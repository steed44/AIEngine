// ============================================================
// pipeline_builder.cpp — 管线构建器
// 根据 PipelineConfig 创建完整的处理器拓扑图：
//   1. 为每个节点创建对应的 IProcessor 实例（模型/预处理/后处理等）
//   2. 按 Edge 配置建立节点间连接
//   3. 标记管线就绪
// ============================================================

#include "config/pipeline_builder.h"
#include "pipeline/pipeline_impl.h"
#include "pipeline/model_node.h"
#include "pipeline/composite_node.h"
#include "pipeline/merge_node.h"
#include "preprocess/resize_node.h"
#include "preprocess/normalize_node.h"
#include "postprocess/nms_node.h"
#include "patchcore/patchcore_node.h"
#include "patchcore/multi_roi_node.h"
#include "backend/backend_factory.h"
#include "engine/thread_pool.h"

namespace aicore {

/**
 * 从配置构建完整管线
 * 遍历 config.nodes 创建处理器实例，再遍历 config.edges 建立有向连接
 * @param config   管线配置
 * @param pipeline 输出参数，构建完成的 IPipeline 对象
 * @param pool     引擎池（用于模型节点的后端复用），可为 nullptr
 * @return Status  构建成功返回空 Status，节点类型/后端未知时返回错误
 */
Status PipelineBuilder::Build(const PipelineConfig& config,
                              std::unique_ptr<IPipeline>& pipeline,
                              std::shared_ptr<EnginePool> pool) {
    auto impl = std::make_unique<PipelineImpl>(pool);

    // 创建线程池，注入 PipelineImpl
    auto threadPool = std::make_shared<ThreadPool>(config.maxConcurrency);
    impl->SetThreadPool(threadPool);

    // 阶段一：遍历所有节点，根据类型创建对应的处理器实例
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
            auto s = backend->Load(info);
            if (!s) return s;
            processor = std::make_shared<ModelNode>(std::move(backend));
        } else if (pc.type == "resize") {
            processor = std::make_shared<ResizeNode>();
        } else if (pc.type == "normalize") {
            processor = std::make_shared<NormalizeNode>();
        } else if (pc.type == "nms") {
            processor = std::make_shared<NmsNode>();
        } else if (pc.type == "merge") {
            processor = std::make_shared<MergeNode>();
        } else if (pc.type == "composite") {
            processor = std::make_shared<CompositeNode>();
        } else if (pc.type == "patchcore") {
            processor = std::make_shared<PatchCoreNode>();
        } else if (pc.type == "multi_roi") {
            processor = std::make_shared<MultiRoiNode>();
        } else {
            return Status{StatusCode::ErrorConfigParse,
                          "unknown node type: " + pc.type};
        }

        auto s = processor->Init(pc.params);
        if (!s) return s;

        // 将线程池注入到支持它的节点
        processor->SetThreadPool(threadPool.get());

        // 将节点注册到管线实现中
        impl->AddNode(pc.id, processor, {});
    }

    // 阶段二：按配置的边建立节点间的有向连接
    for (auto& edge : config.edges) {
        impl->AddEdge(edge.from, edge.to);
    }

    // 标记管线拓扑构建完成，可以开始执行
    impl->MarkReady();
    pipeline = std::move(impl);
    return Status{};
}

} // namespace aicore
