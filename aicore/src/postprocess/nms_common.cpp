// ============================================================
// nms_common.cpp — 非极大值抑制（NMS）通用算法实现
//
// NMS（Non-Maximum Suppression）是目标检测后处理的核心算法，
// 用于去除同一目标上的重复检测框，只保留每个目标的最佳预测。
//
// 算法流程（贪心策略，Greedy NMS）：
//   1. 按置信度降序排列所有候选框
//   2. 选择置信度最高的框，标记为保留
//   3. 遍历其余框，若与当前保留框的 IoU 大于阈值，则抑制（丢弃）
//   4. 对下一个未处理的框重复步骤 2-3，直到所有框都被检查
//
// 时间复杂度：O(n²)
//   外层循环 i 遍历 n 个候选框，内层循环 j 从 i+1 到 n，
//   每对框最多计算一次 IoU，共约 n²/2 次比较。
//   对于检测任务（候选框通常 < 1000），O(n²) 可接受。
//   大模型产出密集预测时可改用 SoftNMS / Fast NMS 等加速变体。
//
// 局限性：
//   - 对重叠目标（人群、密集物体）效果差——同位置不同类目标也可能被抑制
//   - 单纯依赖 IoU 阈值，不区分"框质量"（定位精度 vs 分类置信度）
//   - 严格贪心，不会回溯——若高置信度框定位不准，会错误抑制准确定位的低分框
// ============================================================
#include "postprocess/nms_common.h"
#include <algorithm>

namespace aicore {

// NMS 主函数：对候选框执行贪心非极大值抑制
// [in/out] candidates — 传入候选框列表，函数返回后只保留通过 NMS 的框
//           函数直接修改原 vector，保留框紧凑排列在 vector 前部
// iouThreshold      — IoU 抑制阈值，典型值 0.45~0.5
//                     （高于此值视为重复框并丢弃）
// 算法步骤：
//   1. 置信度排序 → 2. 贪心选择最高分框 → 3. 同标签同类别 IoU 抑制 → 4. 紧凑输出
void NMSCommon(std::vector<NodeResult>& candidates, float iouThreshold) {
    if (candidates.empty()) return;

    // 步骤 1：按置信度降序排列
    // 贪心策略的核心——优先处理高置信度框，用"最强"框抑制"较弱"框
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.confidence > b.confidence; });

    // keep[i] = true 表示第 i 个框保留，false 表示被抑制
    // 初始全部标记为保留，后续在循环中抑制重叠框
    std::vector<bool> keep(candidates.size(), true);
    // 步骤 2-4：贪心抑制循环
    // 外层：遍历每个候选框，跳过已被抑制的框
    // 内层：将当前框与后续所有框比较，同标签且 IoU 超阈值则抑制
    //
    // 为什么同标签比较？
    //   不同类别的目标可能天然重叠（如人牵狗），NMS 通常只在同类间执行。
    //   若要跨类别抑制（如行人检测与车辆检测），需要额外配置。
    //
    // 计算结果：keep[j] = false 表示第 j 个框被丢弃
    // 注意：内层 j 总是从 i+1 开始，因为排序后 i 的置信度高于后续所有框
    for (size_t i = 0; i < candidates.size(); i++) {
        if (!keep[i]) continue;                              // 跳过已被抑制的框
        for (size_t j = i + 1; j < candidates.size(); j++) {
            if (!keep[j] || candidates[i].label != candidates[j].label)
                continue;                                    // 跳过不同标签或已抑制的框
            // 计算 IouBox(候选, 当前保留框) — 若重叠度超阈值，丢弃低分框
            if (IouBox(candidates[i].bbox, candidates[j].bbox) > iouThreshold)
                keep[j] = false;
        }
    }

    // 步骤 5：紧凑重排（in-place compact）
    // 将 keep[i]=true 的框依次移到 vector 前端，丢弃被抑制的框
    // 使用 std::move 避免拷贝，性能优于 erase 逐个删除（O(n) vs O(n²)）
    size_t writeIdx = 0;
    for (size_t i = 0; i < candidates.size(); i++) {
        if (keep[i])
            candidates[writeIdx++] = std::move(candidates[i]);
    }
    candidates.resize(writeIdx);                             // 截断 vector 到实际保留数
}

} // namespace aicore
