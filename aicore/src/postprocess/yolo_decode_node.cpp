// ============================================================
// yolo_decode_node.cpp — YOLO 模型原始张量解码器
//
// 功能：将 YOLO 模型的原始输出张量解码为检测框列表（NodeResult）
//
// 支持的模型版本：
//   - YOLOv8（默认）：anchor-free 架构，使用 DFL（Distribution Focal Loss）
//     解码边界框，每个 grid cell 直接预测 4 个边距分布
//   - YOLOv5/v3（回退）：传统的 anchor-based 架构，直接回归 (cx,cy,w,h)
//
// YOLOv8 anchor-free 解码原理：
//   YOLOv8 抛弃了锚框（anchor boxes），改为每个 grid cell 直接预测：
//   1. 分类分支 (Classification branch)：
//      每个 cell 输出 numClasses 个 sigmoid 得分，取最大值对应类别
//   2. 回归分支 (Regression branch) — DFL 解码：
//      每个 cell 输出 4×regMax = 4×16 = 64 个值，分 4 组每组 16 个
//      每组通过 softmax + 加权求和得到左/上/右/下边距：
//        l = Σ softmax(dist[l,0:15]) * i   （到左边界的距离）
//        t = Σ softmax(dist[t,0:15]) * i   （到上边界的距离）
//        r = Σ softmax(dist[r,0:15]) * i   （到右边界的距离）
//        b = Σ softmax(dist[b,0:15]) * i   （到下边界的距离）
//   3. 从边距还原框坐标（相对于 grid cell 中心 0.5）：
//        x1 = gx + 0.5 - l  →  左上角 x
//        y1 = gy + 0.5 - t  →  左上角 y
//        x2 = gx + 0.5 + r  →  右下角 x
//        y2 = gy + 0.5 + b  →  右下角 y
//
// 坐标映射链（grid → 特征图 → 输入图像 → 原图）：
//   1. grid 坐标 (gx, gy)：当前 cell 在特征图中的行列位置
//   2. gridStride = modelInputSize / gridW：每个 cell 对应输入图像的像素步长
//      例如 640/80=8，即 80×80 特征图每个 cell 对应 8×8 像素区域
//   3. 特征图坐标 → 输入图像坐标：乘以 gridStride
//   4. 输入图像坐标 → 原图坐标：减去 padding，除以 scale（letterbox 逆变换）
// ============================================================
#include "postprocess/yolo_decode_node.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace aicore {

// 初始化：从节点配置中读取 YOLO 解码参数
// 支持参数：
//   confidence_threshold — 类别置信度阈值（低于此值丢弃，默认 0.5）
//   iou_threshold        — NMS IoU 阈值（默认 0.45）
//   num_classes          — 目标类别数（默认 80，COCO）
//   version              — YOLO 版本标识（"v8" 或空，默认 v8）
//   model_input_size     — 模型输入图像尺寸（默认 640）
//   name                 — 节点名称（默认 "yolo_decode"）
Status YoloDecodeNode::Init(const NodeConfig& config) {
    auto it = config.find("confidence_threshold");
    if (it != config.end()) confidenceThreshold_ = std::stof(it->second);
    it = config.find("iou_threshold");
    if (it != config.end()) iouThreshold_ = std::stof(it->second);
    it = config.find("num_classes");
    if (it != config.end()) numClasses_ = std::stoi(it->second);
    it = config.find("version");
    if (it != config.end()) versionStr_ = it->second;
    it = config.find("model_input_size");
    if (it != config.end()) modelInputSize_ = std::stoi(it->second);
    it = config.find("name");
    if (it != config.end()) name_ = it->second;
    return Status{};
}

// 单尺度解码：将一层的 YOLO 输出张量解码为检测框列表
// 输入参数：
//   data     — 模型输出张量的原始 float 指针（CHW 格式，C=regMax*4+numClasses）
//   numBoxes — 该层的总 box 数 = gridH × gridW（特征图所有 cell）
//   stride  — 每个 box 在内存中的跨度 = 通道数 C
//   gridW/H — 特征图的宽/高（如 80×80、40×40、20×20）
//   numClasses — 类别数（COCO=80）
//   scale   — letterbox 缩放因子（用于逆变换到原图坐标）
//   padX/Y  — letterbox 填充像素数（用于逆变换）
//   candidates — [out] 解码后的检测框列表
//
// YOLOv8 张量布局（每个 box 的 stride = C = 4*regMax + numClasses）：
//   [0..15]       — DFL 左边界分布 (l)
//   [16..31]      — DFL 上边界分布 (t)
//   [32..47]      — DFL 右边界分布 (r)
//   [48..63]      — DFL 下边界分布 (b)
//   [64..64+C-1]  — 分类 logits（numClasses 个）
//   合计：4×16 + numClasses = 64 + numClasses 个 float
void YoloDecodeNode::DecodeScale(
    const float* data, int numBoxes, int stride,
    int gridW, int gridH, int numClasses,
    float scale, int padX, int padY,
    std::vector<NodeResult>& candidates) const {

    // regMax = 16：YOLOv8 将每个边距量化为 0~15 共 16 个离散值
    // DFL（Distribution Focal Loss）的核心思想：
    //   不直接回归边距值，而是预测每个离散值上的概率分布，
    //   然后取期望作为最终边距值。这样比直接回归更稳定、精度更高。
    int regMax = 16;
    bool isV8 = (versionStr_ == "v8");

    for (int i = 0; i < numBoxes; i++) {
        // 获取第 i 个 box 的起始指针（偏移 = i × stride 个 float）
        const float* cellData = data + i * stride;
        float cx, cy, w, h;

        if (isV8) {
            // YOLOv8 DFL 解码 lambda：对 16 个离散值做 softmax 加权平均
            // 输入：dist — 长度为 regMax 的原始 logits
            // 输出：加权期望值（边距）
            //
            // softmax 实现（数值稳定版）：
            //   1. 先减最大值防止 exp 溢出（exp(100) 会 overflow）
            //   2. softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
            //   3. 期望 = Σ softmax_i × i
            auto dfl = [&](const float* dist) -> float {
                float maxVal = dist[0];
                for (int j = 1; j < regMax; j++)
                    if (dist[j] > maxVal) maxVal = dist[j];
                float sum = 0, weighted = 0;
                for (int j = 0; j < regMax; j++) {
                    float e = std::exp(dist[j] - maxVal);
                    sum += e;
                    weighted += e * j;
                }
                return weighted / (sum + 1e-9f);
            };

            // 对 4 个方向分别做 DFL 解码，得到边距 (l, t, r, b)
            float l = dfl(cellData);                 // 到左边界的距离
            float t = dfl(cellData + regMax);        // 到上边界的距离
            float r = dfl(cellData + 2 * regMax);    // 到右边界的距离
            float b = dfl(cellData + 3 * regMax);    // 到下边界的距离

            // 计算当前 cell 在 grid 中的位置 (gx, gy)
            // YOLOv8 输出张量按行优先排列：第 i 个 cell 位于第 gy 行第 gx 列
            int gridIdx = i;
            int gx = gridIdx % gridW;
            int gy = gridIdx / gridW;

            // 从边距还原左上/右下角坐标（相对于 grid cell 坐标）
            // grid cell 的左上角为 (gx, gy)，中心为 (gx+0.5, gy+0.5)
            // 边距 l/t/r/b 以 cell 中心为基准向外延伸
            float x1 = gx + 0.5f - l;                // 预测框左边界
            float y1 = gy + 0.5f - t;                // 预测框上边界
            float x2 = gx + 0.5f + r;                // 预测框右边界
            float y2 = gy + 0.5f + b;                // 预测框下边界

            // 从 (x1,y1,x2,y2) 转换为中心点 (cx,cy,w,h) 格式
            // BBox 结构体统一使用中心点格式保存
            cx = (x1 + x2) / 2.0f;                   // 中心 x = (左+右)/2
            cy = (y1 + y2) / 2.0f;                   // 中心 y = (上+下)/2
            w = x2 - x1;                             // 宽度 = 右 - 左
            h = y2 - y1;                             // 高度 = 下 - 上

            // ---- 分类分支：sigmoid 解码 ----
            // 分类 logits 位于 DFL 输出之后，偏移 4*regMax = 64 个 float
            const float* clsLogits = cellData + 4 * regMax;
            float maxScore = 0;
            int bestLabel = 0;
            // 对每个类别的 logit 做 sigmoid 得到概率，取最大值对应类别
            // sigmoid(x) = 1 / (1 + exp(-x))
            // 也可以使用 softmax，但 YOLO 官方使用 sigmoid（支持多标签分类）
            for (int c = 0; c < numClasses; c++) {
                float score = 1.0f / (1.0f + std::exp(-clsLogits[c]));
                if (score > maxScore) {
                    maxScore = score;
                    bestLabel = c;
                }
            }

            // 置信度阈值过滤：跳过低置信度的预测框
            // 这步提前过滤可以减少后续 NMS 的计算量
            if (maxScore < confidenceThreshold_)
                continue;

            // ---- 坐标缩放 1：grid 坐标 → 输入图像坐标 ----
            // gridStride = modelInputSize / gridW（例如 640/80=8）
            // 含义：特征图上 1 个 cell 对应输入图像上的 stride 个像素
            // 乘以 stride 将 grid 坐标映射到 letterbox 填充后的图像坐标
            float gridStride = (float)modelInputSize_ / gridW;
            cx *= gridStride;
            cy *= gridStride;
            w *= gridStride;
            h *= gridStride;

            // ---- 坐标缩放 2：letterbox 逆变换 ----
            // 输入图像经过 letterbox 等比例缩放 + 灰边填充后才送入模型，
            // 解码坐标是以 letterbox 后的图像为参考系的。
            // 现在需要逆变换回原始图像坐标：
            //   1. 减去 padding 偏移（去掉灰边）
            //   2. 除以缩放因子（恢复原始分辨率）
            // 公式：origCoord = (letterboxCoord - pad) / scale
            cx = (cx - padX) / scale;
            cy = (cy - padY) / scale;
            w = w / scale;
            h = h / scale;

            NodeResult det;
            det.label = std::to_string(bestLabel);
            det.confidence = maxScore;
            det.bbox.x = cx;
            det.bbox.y = cy;
            det.bbox.w = w;
            det.bbox.h = h;
            candidates.push_back(std::move(det));

        } else {
            // ---- YOLOv5/v3 兼容路径（anchor-based 直接回归） ----
            // 张量布局：[cx, cy, w, h, cls1, cls2, ..., clsN]
            // 前 4 个值是锚框偏移量：
            //   cx = sigmoid(tx) * 2 - 0.5 + grid_x   （实际公式更复杂，含 anchor 先验）
            //   cy = sigmoid(ty) * 2 - 0.5 + grid_y
            //   w  = pw * exp(tw)^2                    （pw 是锚框宽度）
            //   h  = ph * exp(th)^2                    （ph 是锚框高度）
            // 注意：这里的 cx/cy/w/h 已经是相对于模型输入图像的坐标（含 letterbox），
            // 因此直接减去 padding 除以 scale 即可逆变换回原图坐标
            cx = cellData[0];
            cy = cellData[1];
            w = cellData[2];
            h = cellData[3];

            // 分类分支（与 v8 相同，sigmoid 解码取最大值）
            float maxScore = 0;
            int bestLabel = 0;
            for (int c = 0; c < numClasses; c++) {
                float score = 1.0f / (1.0f + std::exp(-cellData[4 + c]));
                if (score > maxScore) {
                    maxScore = score;
                    bestLabel = c;
                }
            }

            // 置信度阈值过滤
            if (maxScore < confidenceThreshold_)
                continue;

            // letterbox 逆变换到原图坐标
            cx = (cx - padX) / scale;
            cy = (cy - padY) / scale;
            w = w / scale;
            h = h / scale;

            NodeResult det;
            det.label = std::to_string(bestLabel);
            det.confidence = maxScore;
            det.bbox.x = cx;
            det.bbox.y = cy;
            det.bbox.w = w;
            det.bbox.h = h;
            candidates.push_back(std::move(det));
        }
    }
}

// 主处理入口：遍历每个输入帧，解析模型输出的原始张量
//
// 处理流程：
//   1. 从 Frame::roiMap 中读取 letterbox 参数（由前序 LetterboxNode 传入）
//      — letterbox_scale：原图到模型输入的缩放比例
//      — letterbox_pad_x/y：灰边填充量（像素）
//      若无 letterbox 参数（如直接使用 resize 预处理），则 scale=1, pad=0
//   2. 遍历 frame.rawOutputs 中的每个输出张量
//      YOLO 模型通常在 3 个尺度输出（P3/P4/P5，对应 80×80/40×40/20×20），
//      每个尺度调用 DecodeScale 独立解码
//   3. 将所有尺度的候选框合并，保存到输出的 detections 中
//      后续 NmsNode 会对 detections 执行非极大值抑制
Status YoloDecodeNode::Process(const std::vector<Frame>& inputs,
                                std::vector<Frame>& outputs) {
    if (inputs.empty())
        return Status{StatusCode::ErrorInvalidInput, "yolo_decode: no input"};

    for (const auto& frame : inputs) {
        // 从 roiMap 中提取 letterbox 预处理参数
        // 这些参数由上游 LetterboxNode 在预处理阶段写入
        float scale = 1.0f;
        float padX = 0, padY = 0;
        auto itScale = frame.roiMap.find("letterbox_scale");
        if (itScale != frame.roiMap.end()) scale = itScale->second;
        auto itPadX = frame.roiMap.find("letterbox_pad_x");
        if (itPadX != frame.roiMap.end()) padX = itPadX->second;
        auto itPadY = frame.roiMap.find("letterbox_pad_y");
        if (itPadY != frame.roiMap.end()) padY = itPadY->second;

        std::vector<NodeResult> allCandidates;

        // 遍历多尺度输出：YOLO 模型输出通常包含 3 个尺度
        // P3：大特征图（浅层），检测小目标，grid 细密（如 80×80）
        // P4：中特征图（中层），检测中目标（如 40×40）
        // P5：小特征图（深层），检测大目标，grid 稀疏（如 20×20）
        for (const auto& tensor : frame.rawOutputs) {
            // 有效输出需满足：4 维张量 [N, C, H, W]，batch N=1
            if (tensor.shape.size() != 4 || tensor.shape[0] != 1)
                continue;
            if (!tensor.data || tensor.bytes == 0)
                continue;

            int C = (int)tensor.shape[1];  // 通道数 = 4*regMax + numClasses
            int H = (int)tensor.shape[2];  // 特征图高度（grid 行数）
            int W = (int)tensor.shape[3];  // 特征图宽度（grid 列数）
            int numBoxes = H * W;          // 总 grid cell 数
            int stride = C;                // 每个 cell 的 float 数 = 通道数

            const float* data = static_cast<const float*>(tensor.data);
            DecodeScale(data, numBoxes, stride, W, H,
                        numClasses_, scale, (int)padX, (int)padY,
                        allCandidates);
        }

        // 组装输出帧：包含所有尺度解码后的候选框
        Frame out;
        out.frameId = frame.frameId;
        out.timestamp = frame.timestamp;
        out.sourceId = frame.sourceId;
        out.detections = std::move(allCandidates);
        outputs.push_back(std::move(out));
    }
    return Status{};
}

std::string YoloDecodeNode::GetName() const { return name_; }
std::string YoloDecodeNode::GetType() const { return "yolo_decode"; }

} // namespace aicore
