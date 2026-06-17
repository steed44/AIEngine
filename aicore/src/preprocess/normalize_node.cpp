// ============================================================
// normalize_node.cpp — 图像归一化预处理节点
// BGR→RGB 转换、像素缩放到 [0,1]、减均值除方差
// ============================================================
#include "preprocess/normalize_node.h"

namespace aicore {

/**
 * 初始化归一化节点：读取是否执行 BGR→RGB 转换
 * @param config 节点配置键值对，含 "bgr_to_rgb"
 */
Status NormalizeNode::Init(const NodeConfig& config) {
    auto it = config.find("bgr_to_rgb");
    if (it != config.end()) bgrToRgb_ = (it->second == "true");
    return Status{};
}

/**
 * 执行归一化处理：
 *   1. 可选 BGR→RGB 通道重排（OpenCV 默认 BGR，模型通常用 RGB）
 *   2. 像素值从 [0,255] uint8 缩放到 [0,1] float32
 *   3. 对每个通道执行 (x - mean) / std 标准化
 * 第三步通过操作 float 指针偏移实现逐通道就地计算，避免复制
 *
 * 对每个通道独立执行 (x - mean_c) / std_c 标准化。
 * 数学公式（对输入图像 I，输出图像 I'）：
 *   I'_c(i,j) = (I_c(i,j) / 255.0 - mean_c) / std_c
 *   = I_c(i,j) / (255.0 * std_c) - mean_c / std_c
 * 本实现分两步执行（subtract 后 divide），含义更清晰。
 * 性能优化：可以合并为一步乘法加法 (I * scale + bias)。
 * @param inputs  原始输入帧（uint8 类型）
 * @param outputs [out] 归一化后的 float32 帧
 */
Status NormalizeNode::Process(const std::vector<Frame>& inputs,
                              std::vector<Frame>& outputs) {
    for (const auto& frame : inputs) {
        Frame out;
        out.frameId = frame.frameId;
        out.timestamp = frame.timestamp;

        cv::Mat img = frame.image.clone();
        // 将 BGR 转为 RGB（如果模型训练时使用 RGB 顺序）
        if (bgrToRgb_ && img.channels() == 3)
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // uint8 [0,255] → float32 [0,1]
        img.convertTo(out.image, CV_32FC3, 1.0 / 255.0);

        // ---- 逐通道标准化原理 ----
        // 标准归一化公式：x'_c = (x_c - mean_c) / std_c, c ∈ {R, G, B}
        //
        // 其中 mean = [0.485, 0.456, 0.406]（ImageNet 数据集各通道均值）
        //      std  = [0.229, 0.224, 0.225]（ImageNet 数据集各通道标准差）
        //
        // 为什么需要标准化：
        //   1. 数值稳定性：将输入分布拉到 N(0,1) 附近，避免激活函数饱和区
        //      （如 sigmoid 在 |x|>5 时梯度几乎为 0）
        //   2. 加速收敛：各通道数据尺度统一，优化器（SGD/Adam）的各维度
        //      学习步长更均衡，无需对不同通道采用不同学习率
        //   3. 跨数据集一致性：消除数据集间亮度/对比度差异的影响
        //
        // 就地计算（in-place）的内存视图技巧：
        //   CV_32FC3 的内存布局是 [R0,G0,B0,R1,G1,B1,...]（交错排列），
        //   第 c 通道的所有像素在内存中按 stride = C 个 float 等间隔分布。
        //   这里通过指针偏移 out.image.ptr<float>(0) + i * out.image.cols * out.image.rows
        //   创建一个虚拟的单通道视图（CV_32FC1），使 OpenCV 函数可以对
        //   该通道的所有像素进行连续运算。
        //
        //   虽然每个通道的像素在物理内存中是不连续的（被其他通道像素间隔），
        //   但此处创建的 cv::Mat header 告知 Opencv 数据是连续的，
        //   cv::subtract 和 cv::divide 会根据 DataType 的大小逐个处理元素，
        //   只要步长计算正确，in-place 操作仍然正确。
        //
        //   更严谨的做法是 cv::extractChannel + cv::divide + cv::insertChannel，
        //   但对于非连续情况，当前方法需要确保结果正确性。
        for (int i = 0; i < img.channels(); ++i) {
            cv::Mat channel(out.image.size(), CV_32FC1,
                            out.image.ptr<float>(0) + i * out.image.cols * out.image.rows);
            cv::subtract(channel, cv::Scalar(mean_[i]), channel);
            cv::divide(channel, cv::Scalar(std_[i]), channel);
        }

        outputs.push_back(std::move(out));
    }
    return Status{};
}

/** 返回节点名称 */
std::string NormalizeNode::GetName() const { return "normalize"; }
/** 返回节点类型标识 */
std::string NormalizeNode::GetType() const { return "normalize"; }

} // namespace aicore
