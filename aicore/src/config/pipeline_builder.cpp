// ============================================================
// pipeline_builder.cpp — 管线构建器
// 根据 PipelineConfig 创建完整的处理器拓扑图。
// 新增节点类型时在 Build() 的 switch 中添加分支即可。
// ============================================================

#include "config/pipeline_builder.h"
#include "pipeline/pipeline_impl.h"
#include "pipeline/model_node.h"
#include "pipeline/composite_node.h"
#include "pipeline/merge_node.h"
#include "preprocess/resize_node.h"
#include "preprocess/normalize_node.h"
#include "postprocess/nms_node.h"
#include "pipeline/fusion_node.h"
#include "preprocess/letterbox_node.h"
#include "postprocess/yolo_decode_node.h"
#include "patchcore/patchcore_node.h"
#include "patchcore/multi_roi_node.h"
#include "backend/backend_factory.h"
#include "engine/engine_pool.h"
#include "engine/thread_pool.h"

namespace aicore {

namespace {

// ============================================================
// 简单节点工厂（除 model 外所有类型）
// 通过 constexpr map 将类型名 → creator lambda，避免 if-else 链膨胀。
// 新增简单节点类型时在此 map 加一行即可。
// ============================================================
std::shared_ptr<IProcessor> CreateSimpleNode(const std::string& type) {
    static const std::unordered_map<std::string,
                                     std::shared_ptr<IProcessor>(*)()> kFactories = {
        {"resize",      []() -> std::shared_ptr<IProcessor> { return std::make_shared<ResizeNode>(); }},
        {"normalize",   []() -> std::shared_ptr<IProcessor> { return std::make_shared<NormalizeNode>(); }},
        {"letterbox",   []() -> std::shared_ptr<IProcessor> { return std::make_shared<LetterboxNode>(); }},
        {"yolo_decode", []() -> std::shared_ptr<IProcessor> { return std::make_shared<YoloDecodeNode>(); }},
        {"nms",         []() -> std::shared_ptr<IProcessor> { return std::make_shared<NmsNode>(); }},
        {"merge",       []() -> std::shared_ptr<IProcessor> { return std::make_shared<MergeNode>(); }},
        {"composite",   []() -> std::shared_ptr<IProcessor> { return std::make_shared<CompositeNode>(); }},
        {"patchcore",   []() -> std::shared_ptr<IProcessor> { return std::make_shared<PatchCoreNode>(); }},
        {"multi_roi",   []() -> std::shared_ptr<IProcessor> { return std::make_shared<MultiRoiNode>(); }},
        {"fusion",      []() -> std::shared_ptr<IProcessor> { return std::make_shared<FusionNode>(); }},
    };
    auto it = kFactories.find(type);
    return it != kFactories.end() ? it->second() : nullptr;
}

} // namespace

/**
 * 从配置构建完整管线
 * 遍历 config.nodes 创建节点，遍历 config.edges 建立有向连接。
 */
Status PipelineBuilder::Build(const PipelineConfig& config,
                              std::unique_ptr<IPipeline>& pipeline,
                              std::shared_ptr<EnginePool> pool) {
    auto impl = std::make_unique<PipelineImpl>(pool);

    auto threadPool = std::make_shared<ThreadPool>(config.maxConcurrency);
    impl->SetThreadPool(threadPool);

    // 阶段一：按节点类型创建所有处理器
    // model 节点有 Backend 创建/Load 特殊逻辑，单独处理。
    // 其余节点统一走 CreateSimpleNode 工厂 map。
    for (auto& pc : config.nodes) {
        std::shared_ptr<IProcessor> processor;

        if (pc.type == "model") {
            auto backend = BackendFactory::Create(pc.backend);
            if (!backend)
                return Status{StatusCode::ErrorConfigParse,
                              "unknown backend for " + pc.id};
            ModelInfo info;
            info.modelPath = pc.modelPath;
            info.backend = pc.backend;
            info.deviceId = pc.deviceId;
            info.batchSize = pc.batchSize;
            Status s = backend->Load(info);
            if (!s) return s;
            processor = std::make_shared<ModelNode>(std::move(backend));
        } else {
            processor = CreateSimpleNode(pc.type);
            if (!processor)
                return Status{StatusCode::ErrorConfigParse,
                              "unknown node type: " + pc.type};
        }

        Status s = processor->Init(pc.params);
        if (!s) return s;

        processor->SetThreadPool(threadPool.get());
        impl->AddNode(pc.id, processor, {});
    }

    // 阶段二：按配置的边建立连接
    for (auto& edge : config.edges) {
        impl->AddEdge(edge.from, edge.to);
    }

    impl->MarkReady();
    pipeline = std::move(impl);
    return Status{};
}

} // namespace aicore
