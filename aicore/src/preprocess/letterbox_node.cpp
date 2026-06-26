// ============================================================
// letterbox_node.cpp — Letterbox 等比例缩放 + 灰边填充
//
// Letterbox 算法（YOLO 系列标准预处理）：
//
// 为什么需要 Letterbox？
//   目标检测模型要求固定尺寸输入（如 640×640），
//   而原始图像分辨率各异（1920×1080、800×600 等）。
//   - 直接拉伸（resize）会改变目标长宽比，影响检测精度
//   - 直接裁切（crop）可能丢失目标
//   Letterbox 保持长宽比缩放 + 灰边填充，是检测任务的最佳实践
//
// 算法步骤：
//   1. 计算等比例缩放因子 scale = min(targetW/w, targetH/h)
//      保证缩放后图像完全容纳在 target 尺寸内
//   2. 将原图缩放到 (w*scale, h*scale)
//   3. 计算 padding，使缩放后的图像在 target 中居中
//      padX = (targetW - newW) / 2
//      padY = (targetH - newH) / 2
//   4. 用灰边填充四周，补齐到 target 尺寸
//
// 为什么用填充色 (114,114,114)？
//   这是 ImageNet 数据集在 BGR 空间下的均值像素值。
//   YOLO 官方实现（ultralytics）默认使用 (114,114,114) 作为填充色，
//   使得填充区域在数值上接近自然图像的统计均值，
//   不会引入额外干扰信号。
//
// 为什么 padding 对齐到 32 像素？
//   YOLO 模型的下采样倍数通常为 32（5 个 stride=2 的卷积层）。
//   保证输入尺寸是 32 的倍数，可以避免 TensorRT/ONNX Runtime 中
//   因尺寸不对齐导致的隐式 padding 或性能下降。
//   不过本实现不要求 32 对齐——调用方应自行确保 targetWidth/Height
//   是 32 的倍数（如 640、672、704）。
//
// 坐标映射（用于后续解码）：
//   模型输出的检测框坐标是以 letterbox 填充后的图像为参考系的。
//   解码时需要进行逆变换（在 YoloDecodeNode 中实现）：
//     origX = (letterboxX - padX) / scale
//     origY = (letterboxY - padY) / scale
//   scale, padX, padY 通过 roiMap 传递给下游节点。
// ============================================================
#include "preprocess/letterbox_node.h"

namespace aicore {

// 初始化：从节点配置读取 letterbox 参数
// 支持参数：
//   width      — 目标宽度，默认 640
//   height     — 目标高度，默认 640
//   pad_color  — 填充色，BGR 格式如 "114,114,114"
Status LetterboxNode::Init(const NodeConfig& config) {
    auto it = config.find("width");
    if (it != config.end()) targetWidth_ = std::stoi(it->second);
    it = config.find("height");
    if (it != config.end()) targetHeight_ = std::stoi(it->second);
    it = config.find("pad_color");
    if (it != config.end()) {
        // pad_color 格式: "114,114,114" (BGR)
        // sscanf_s 按照 B,G,R 顺序解析
        int r, g, b;
        sscanf_s(it->second.c_str(), "%d,%d,%d", &b, &g, &r);
        padColor_ = cv::Scalar(b, g, r);
    }
    return Status{};
}

// 预处理主函数：对每帧执行 letterbox 操作
// 输入：任意尺寸的原始图像
// 输出：等比例缩放 + 灰边填充后的固定尺寸图像
//
// 数学公式：
//   给定原图 (w, h)，目标 (tw, th)：
//     scale = min(tw/w, th/h)
//     newW = round(w * scale)
//     newH = round(h * scale)
//     padX = (tw - newW) / 2
//     padY = (th - newH) / 2
//
//   变换矩阵（仿射变换等价形式）：
//     [x_out]   [scale  0      padX] [x_in ]
//     [y_out] = [0      scale  padY] [y_in ]
//     [1    ]   [0      0      1   ] [1    ]
//   scale 是线性缩放部分，padX/padY 是平移补偿部分
Status LetterboxNode::Process(const std::vector<Frame>& inputs,
                                std::vector<Frame>& outputs) {
    if (inputs.empty())
        return Status{StatusCode::ErrorInvalidInput, "letterbox: no input"};

    for (const auto& frame : inputs) {
        const auto& img = frame.image;
        int h = img.rows, w = img.cols;
        if (h == 0 || w == 0)
            return Status{StatusCode::ErrorPreprocess, "letterbox: empty image"};

        // 步骤 1：计算等比例缩放因子
        // 取宽和高中较小的缩放比，保证缩放后图像完全落在目标尺寸内
        // 例如原图 1920×1080，目标 640×640：
        //   scale_x = 640/1920 ≈ 0.333
        //   scale_y = 640/1080 ≈ 0.593
        //   scale = min(0.333, 0.593) = 0.333
        //   缩放后尺寸：640×360（宽度刚好占满，高度不足）
        float scale = std::min((float)targetWidth_ / w, (float)targetHeight_ / h);
        int newW = (int)std::round(w * scale);
        int newH = (int)std::round(h * scale);

        // 步骤 2：等比例缩放（双线性插值）
        // 保持长宽比不变，只改变图像分辨率
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

        // 步骤 3：计算 padding 偏移量
        // 将缩放后的图像居中放置，左右/上下均匀填充灰边
        // 由于 newW ≤ targetWidth，newH ≤ targetHeight，padding 总为非负
        // 整数除法向下取整，最终图像可能偏离中心 1 像素（可忽略）
        int padX = (targetWidth_ - newW) / 2;
        int padY = (targetHeight_ - newH) / 2;

        // 步骤 4：灰边填充
        // 在上/下/左/右四个方向添加相同像素的填充
        // 注意：OpenCV 的 copyMakeBorder 的 top/bottom 和 left/right
        // 是独立参数，这里上下相同 (padY)，左右相同 (padX)
        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, padY, padY, padX, padX,
                           cv::BORDER_CONSTANT, padColor_);

        Frame out(std::move(padded), frame.frameId);
        // 传递 letterbox 参数给下游解码节点（YoloDecodeNode）
        // 这些参数用于将模型输出的检测框坐标逆变换回原图坐标
        out.roiMap["letterbox_scale"] = scale;     // 缩放因子
        out.roiMap["letterbox_pad_x"] = (float)padX;  // X 方向填充量
        out.roiMap["letterbox_pad_y"] = (float)padY;  // Y 方向填充量
        outputs.push_back(std::move(out));
    }
    return Status{};
}

std::string LetterboxNode::GetName() const { return "letterbox"; }
std::string LetterboxNode::GetType() const { return "letterbox"; }

} // namespace aicore
