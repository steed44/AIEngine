#include "config/pipeline_builder.h"
#include "pipeline/pipeline_impl.h"
#include "pipeline/model_node.h"
#include "pipeline/composite_node.h"
#include "pipeline/merge_node.h"
#include "preprocess/resize_node.h"
#include "preprocess/normalize_node.h"
#include "postprocess/nms_node.h"
#include "patchcore/patchcore_node.h"
#include "backend/backend_factory.h"

namespace aicore {

Status PipelineBuilder::Build(const PipelineConfig& config,
                              std::unique_ptr<IPipeline>& pipeline,
                              std::shared_ptr<EnginePool> pool) {
    auto impl = std::make_unique<PipelineImpl>(pool);

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
            processor = std::make_shared<ModelNode>(std::shared_ptr<IModelBackend>(std::move(backend)));
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
        } else {
            return Status{StatusCode::ErrorConfigParse,
                          "unknown node type: " + pc.type};
        }

        auto s = processor->Init(pc.params);
        if (!s) return s;

        impl->AddNode(pc.id, processor, {});
    }

    for (auto& edge : config.edges) {
        impl->AddEdge(edge.from, edge.to);
    }

    impl->MarkReady();
    pipeline = std::move(impl);
    return Status{};
}

} // namespace aicore
