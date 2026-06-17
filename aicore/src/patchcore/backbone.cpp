// ============================================================
// backbone.cpp — IBackbone 工厂方法实现
// 功能：根据类型字符串创建对应的 Backbone 实例，是 backbone
//       体系的总入口点，供外部代码统一调用
// ============================================================
#include "patchcore/backbone.h"
#include "patchcore/backbone_opencv.h"
#include "patchcore/backbone_model.h"
#ifdef AICORE_HAS_LIBTORCH
#include "patchcore/backbone_libtorch.h"
#endif

namespace aicore {

// -------------------------------------------------------
// 创建 Backbone 实例的工厂函数
// 支持的 type 值：
//   - "opencv_dnn"   → OpenCVDnnBackbone（默认，纯 OpenCV 依赖）
//   - "model_backend" → ModelBackendBackbone（通过通用后端接口）
//   - "libtorch"     → LibTorchBackbone（需要 AICORE_HAS_LIBTORCH 宏）
// @param type   backbone 类型字符串
// @param config 配置参数（透传给具体实现的 Init 方法）
// @return 创建的 IBackbone 实例，类型未知时返回 nullptr
// -------------------------------------------------------
std::unique_ptr<IBackbone> CreateBackbone(const std::string& type, const NodeConfig& config) {
    if (type == "opencv_dnn") {
        return std::make_unique<OpenCVDnnBackbone>();
    } else if (type == "model_backend") {
        return std::make_unique<ModelBackendBackbone>();
    }
#ifdef AICORE_HAS_LIBTORCH
    else if (type == "libtorch") {
        return std::make_unique<LibTorchBackbone>();
    }
#endif
    return nullptr;
}

} // namespace aicore
