// ============================================================
// backbone_libtorch.cpp — LibTorch Backbone 实现
// 功能：使用 PyTorch JIT 加载 TorchScript 模型，执行推理
//       并按照 ImageNet 标准化预处理图像
// 注意：整个文件受 AICORE_HAS_LIBTORCH 宏控制，未定义时
//       所有方法返回错误或空结果
// ============================================================
#include "patchcore/backbone_libtorch.h"
#include <opencv2/imgproc.hpp>
#include <sstream>

namespace aicore {

// 将逗号分隔的层名字符串解析为字符串向量
// 例如 "layer2,layer3" → ["layer2", "layer3"]
static std::vector<std::string> SplitLayerNames(const std::string& s) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) result.push_back(item);
    }
    return result;
}

// -------------------------------------------------------
// 初始化 LibTorch backbone：加载 TorchScript 模型并设 eval 模式
// 配置参数：
//   - model_path:        TorchScript .pt 文件路径（必需）
//   - backbone_layers:   待提取的中间层名（逗号分隔，可选）
//   - input_size:        输入图像缩放尺寸（可选，默认 224）
// -------------------------------------------------------
Status LibTorchBackbone::Init(const NodeConfig& config) {
#ifdef AICORE_HAS_LIBTORCH
    auto it = config.find("model_path");
    if (it == config.end()) {
        return Status{StatusCode::ErrorConfigParse, "libtorch: model_path required"};
    }
    try {
        // 加载 TorchScript 模型并切换到推理模式
        module_ = torch::jit::load(it->second);
        module_.eval();
    } catch (const std::exception& e) {
        return Status{StatusCode::ErrorModelLoad,
            std::string("libtorch: failed to load model: ") + e.what()};
    }

    auto layers = config.find("backbone_layers");
    if (layers != config.end()) {
        outputLayerNames_ = SplitLayerNames(layers->second);
    }

    auto is = config.find("input_size");
    if (is != config.end()) inputSize_ = std::stoi(is->second);

    return Status{};
#else
    (void)config;
    return Status{StatusCode::ErrorModelLoad,
        "libtorch: requires AICORE_HAS_LIBTORCH"};
#endif
}

// -------------------------------------------------------
// 提取特征：预处理 → Tensor 转换 → 模型推理 → PatchFeature 列表
// 预处理步骤：resize → BGR→RGB → float32/255 → NHWC→NCHW → ImageNet 标准化
// 输出遍历：每层特征图的每个空间位置(HxW)生成一个 PatchFeature，
//          每个位置的 channels 维向量作为特征描述
// -------------------------------------------------------
std::vector<PatchFeature> LibTorchBackbone::Extract(const cv::Mat& image) {
#ifdef AICORE_HAS_LIBTORCH
    // ---- 图像预处理流程 ----
    // 预处理步骤遵循 ImageNet 标准，保证输入分布与 backbone 预训练一致：
    //   1. resize: 缩放到统一尺寸（默认 224×224），保证特征图尺寸确定
    //   2. BGR→RGB: OpenCV 读图是 BGR 顺序，PyTorch 预训练模型用 RGB
    //   3. uint8→float32: 像素值从 [0,255] 映射到 [0,1]
    //   4. HWC→CHW (permute): OpenCV 是 HWC 布局，PyTorch 需要 CHW
    //   5. ImageNet 标准化: (x - mean) / std，mean=[0.485,0.456,0.406]
    //      std=[0.229,0.224,0.225]，使输入分布接近 N(0,1)
    //      标准化原因：CNN 权重训练时假设输入符合此分布
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputSize_, inputSize_));
    cv::Mat rgb, floatImg;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(floatImg, CV_32F, 1.0 / 255);

    // ---- HWC → CHW + ImageNet 标准化 ----
    auto tensor = torch::from_blob(floatImg.data, {1, inputSize_, inputSize_, 3}, torch::kFloat);
    tensor = tensor.permute({0, 3, 1, 2});
    tensor[0][0] = tensor[0][0].sub_(0.485).div_(0.229);
    tensor[0][1] = tensor[0][1].sub_(0.456).div_(0.224);
    tensor[0][2] = tensor[0][2].sub_(0.406).div_(0.225);

    // ---- 模型推理 ----
    // NoGradGuard 禁用梯度计算：
    //   推理时不需要梯度，禁用后可显著减少显存占用和计算量
    //   内存节省：推理仅需存储前向激活值，无需存储中间计算图
    torch::NoGradGuard noGrad;
    auto output = module_.forward({tensor});

    // ---- 解析输出：支持两种格式 ----
    // 格式 A (推荐，由 export_patchcore_backbone.py 导出):
    //   模型返回 (feat_layer1, feat_layer2, ...) 元组，
    //   每个元素是形状为 [1, C, H, W] 的中间层特征图
    //
    // 格式 B (兼容模式): 模型返回单个 tensor，
    //   适用于未修改的标准 TorchScript 模型（如分类任务的 .pt 文件）
    std::vector<torch::Tensor> layerTensors;

    if (output.isTuple()) {
        // 格式 A: 多输出元组
        auto tuple = output.toTuple();
        int numLayers = static_cast<int>(tuple->elements().size());
        int namedLayers = static_cast<int>(outputLayerNames_.size());
        int count = (namedLayers > 0) ? std::min(numLayers, namedLayers) : numLayers;
        for (int i = 0; i < count; i++) {
            layerTensors.push_back(tuple->elements()[i].toTensor());
        }
    } else if (output.isTensor()) {
        // 格式 B: 单 tensor 输出，当作单层特征处理
        layerTensors.push_back(output.toTensor());
    } else {
        // 格式不识别，返回空
        return {};
    }

    // ---- 从特征图张量提取 PatchFeature 列表 ----
    // 对每层特征图（如 layer3 输出 [1, 512, 28, 28]）：
    //   - 将 Tensor 拷贝到 CPU 并确保内存连续
    //   - 遍历所有空间位置 (H × W)
    //   - 每个位置提取 channels 维向量作为 PatchFeature
    //   - 记录 layerIdx/patchRow/patchCol 以便后续空间定位
    //
    // CHW 布局的内存访问模式：
    //   data[c][h][w] = data[(c * H + h) * W + w]
    //   外层循环 row, 内层 col 保证了缓存友好（同一行的不同列在内存中连续）
    std::vector<PatchFeature> features;
    for (int li = 0; li < static_cast<int>(layerTensors.size()); li++) {
        auto feat = layerTensors[li].cpu().contiguous();
        int channels = feat.size(1);
        int h = feat.size(2);
        int w = feat.size(3);
        float* data = feat.data_ptr<float>();

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                PatchFeature pf;
                pf.layerIdx = li;
                pf.patchRow = row;
                pf.patchCol = col;
                pf.features.resize(channels);
                // 按 CHW 布局读取每个通道在该位置的值
                for (int c = 0; c < channels; c++)
                    pf.features[c] = data[(c * h + row) * w + col];
                features.push_back(pf);
            }
        }
    }
    return features;
#else
    (void)image;
    return {};
#endif
}

} // namespace aicore
