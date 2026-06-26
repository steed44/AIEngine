// ============================================================
// libtorch_backend.cpp — LibTorch 后端实现
// 已内联至 backend_factory.cpp 以解决跨翻译单元链接问题
// 保留此文件作为未来独立编译的占位
//
// 设计决策：LibTorch 后端实现在 backend_factory.cpp 中内联（#include 方式），
//   而非在此文件中单独编译。原因：
//   1. backend_factory.cpp 的 BackendFactory::Create() 需要调用 CreateLibTorchBackend()
//   2. 如果此文件独立编译，CreateLibTorchBackend() 符号在 aicore.dll 中可见
//   3. 但 backend_factory.cpp 引用此符号时，静态链接可能导致符号不可见
//   4. 内联后所有代码在同一翻译单元中，无符号链接问题
//
// 未来重构：
//   如果 LibTorch 后端变得复杂，应考虑将 CreateLibTorchBackend() 移到单独的 .cpp，
//   并正确设置 DLL 导出/导入声明
// ============================================================
