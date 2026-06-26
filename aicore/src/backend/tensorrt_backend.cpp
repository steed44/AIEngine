// ============================================================
// tensorrt_backend.cpp — TensorRT 后端占位
// TensorRT 实现尚未完成，保留此文件供未来扩展
//
// 计划中的实现流程：
//   Load():
//     1. 读取 .engine 序列化引擎文件
//     2. nvinfer1::createInferRuntime(logger) 创建运行时
//     3. runtime->deserializeCudaEngine(modelData, size) 反序列化引擎
//     4. engine->createExecutionContext() 创建执行上下文
//     5. 查询输入输出 binding 的名称、形状和数据类型
//
//   Infer():
//     1. 将输入 Tensor 数据拷贝到 GPU binding buffer
//     2. context->enqueueV2(bindings, stream, nullptr) 异步执行
//     3. 将输出从 GPU buffer 拷贝到 CPU 输出 Tensor
//
// 当前状态：Stub 实现（TensorRTBackendStub 在 backend_factory.cpp 中），
//   Load() 返回成功，Infer() 返回"未实现"错误。
//   在 CUDA + TensorRT SDK 环境下，应将此文件替换为完整实现。
// ============================================================
