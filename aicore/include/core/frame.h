// 帧数据结构定义
// 表示 pipeline 中传递的一帧图像数据，包含图像内容、时间戳和 ROI 信息
//
// 数据流：pipeline 中每个 IProcessor 节点接收 Frame 向量，处理后输出 Frame 向量。
//   Frame 在一个节点的输出成为下一个节点的输入，沿 DAG 边传递。
//   节点的 Process(inputs → outputs) 可以追加新的 Frame 或修改现有 Frame。
//
// 所有权：Frame 的 image 通过 cv::Mat 自动管理（引用计数），
//   rawOutputs 中的 Tensor data 由后端分配，需调用方释放。
#pragma once
#include "core/types.h"
#include <opencv2/core.hpp>
#include <chrono>
#include <string>

// aicore 命名空间，包含 AI 引擎核心的所有类型和接口
namespace aicore {

// 表示一帧图像及其附属元数据
// 作为 pipeline 中处理器节点之间的数据传递单元
struct Frame {
    cv::Mat image;                            // OpenCV 图像矩阵，存储实际像素数据（BGR 格式）
    uint64_t frameId = 0;                     // 帧序号，单调递增，用于追踪和排序
    uint64_t timestamp = 0;                   // 创建时间戳（毫秒精度，steady_clock）
    std::string sourceId;                     // 来源标识（摄像头 ID、视频文件路径或图像文件名）
    std::map<std::string, float> roiMap;      // ROI（感兴趣区域）名称到置信度/分数的映射
    std::vector<NodeResult> detections;        // 检测结果列表（NMS 等后处理节点填充）
    std::vector<Tensor> rawOutputs;            // 模型原始输出张量列表（ModelNode 写入，解码节点消费）

    Frame() = default;

    // 构造带图像和可选 ID 的帧，自动记录当前时间戳
    // @param img 图像数据（通过 move 语义传入）
    // @param id  帧序号，默认为 0
    Frame(cv::Mat img, uint64_t id = 0)
        : image(std::move(img)), frameId(id) {
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // 判断帧是否为空（无有效图像数据）
    // @return true 表示图像为空
    bool empty() const { return image.empty(); }
    // 获取图像宽度（像素）
    // @return 图像列数
    int width() const { return image.cols; }
    // 获取图像高度（像素）
    // @return 图像行数
    int height() const { return image.rows; }
};

} // namespace aicore
