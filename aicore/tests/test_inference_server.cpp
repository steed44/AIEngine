#include <gtest/gtest.h>
#include "server/inference_server.h"
#include <opencv2/imgproc.hpp>

using namespace aicore;

class InferenceServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 确保每次测试前服务器状态干净
        InferenceServer::Instance().Shutdown();
    }

    void TearDown() override {
        InferenceServer::Instance().Shutdown();
    }

    cv::Mat MakeTestImage(int w = 224, int h = 224) {
        return cv::Mat(h, w, CV_8UC3, cv::Scalar(128, 128, 128));
    }
};

// 测试：LoadModel 加载 stub 后端成功
TEST_F(InferenceServerTest, LoadModelSucceeds) {
    auto& server = InferenceServer::Instance();
    auto s = server.LoadModel("test_model", "dummy.onnx", "onnxruntime", 512, 1);
    EXPECT_TRUE(s);
}

// 测试：LoadModel 未知后端字符串 fallback 到 ONNXRuntime（当前行为）
TEST_F(InferenceServerTest, LoadModelUnknownBackendFallback) {
    auto& server = InferenceServer::Instance();
    auto s = server.LoadModel("bad_model", "dummy.onnx", "nonexistent", 512, 1);
    // 当前实现：未知字符串默认到 ONNXRuntime，返回成功
    EXPECT_TRUE(s);
}

// 测试：IsLoaded 返回 true（加载后）
TEST_F(InferenceServerTest, IsLoadedAfterLoad) {
    auto& server = InferenceServer::Instance();
    server.LoadModel("loaded_model", "dummy.onnx", "onnxruntime", 256, 1);
    EXPECT_TRUE(server.IsLoaded("loaded_model"));
}

// 测试：IsLoaded 对于未加载模型返回 false
TEST_F(InferenceServerTest, IsLoadedForUnknownReturnsFalse) {
    auto& server = InferenceServer::Instance();
    EXPECT_FALSE(server.IsLoaded("unknown"));
}

// 测试：ListModels 包含已加载模型名称
TEST_F(InferenceServerTest, ListModelsContainsLoadedModel) {
    auto& server = InferenceServer::Instance();
    server.LoadModel("list_test", "dummy.onnx", "onnxruntime", 128, 1);

    std::string models = server.ListModels();
    EXPECT_NE(models.find("list_test"), std::string::npos);
}

// 测试：InferSync 在 stub 后端返回错误（stub 不支持推理）
TEST_F(InferenceServerTest, InferSyncReturnsErrorFromStub) {
    auto& server = InferenceServer::Instance();
    server.LoadModel("stub_model", "dummy.onnx", "onnxruntime", 256, 1);

    std::vector<cv::Mat> inputs = {MakeTestImage()};
    std::vector<cv::Mat> outputs;
    auto s = server.InferSync(inputs, outputs, "stub_model");
    // stub 返回 ErrorInternal（"stub cannot infer"）
    EXPECT_FALSE(s);
    EXPECT_EQ(s.code, StatusCode::ErrorInternal);
}

// 测试：InferSync 未加载模型返回错误
TEST_F(InferenceServerTest, InferSyncUnloadedModelReturnsError) {
    auto& server = InferenceServer::Instance();
    std::vector<cv::Mat> inputs = {MakeTestImage()};
    std::vector<cv::Mat> outputs;
    auto s = server.InferSync(inputs, outputs, "not_loaded");
    EXPECT_FALSE(s);
    EXPECT_EQ(s.code, StatusCode::ErrorModelLoad);
}

// 测试：InferAsync 未加载模型返回错误
TEST_F(InferenceServerTest, InferAsyncUnloadedModelReturnsError) {
    auto& server = InferenceServer::Instance();
    InferenceRequest req;
    req.modelName = "not_loaded";
    req.inputs = {MakeTestImage()};
    req.callback = [](StatusCode, std::vector<cv::Mat>, const std::string&) {};

    auto s = server.InferAsync(std::move(req));
    EXPECT_FALSE(s);
}

// 测试：InferAsync 回调被调用（即使返回错误）
TEST_F(InferenceServerTest, InferAsyncCallbackCalledOnStub) {
    auto& server = InferenceServer::Instance();
    server.LoadModel("async_model", "dummy.onnx", "onnxruntime", 256, 1);

    bool called = false;
    StatusCode resultCode = StatusCode::OK;
    InferenceRequest req;
    req.modelName = "async_model";
    req.inputs = {MakeTestImage()};
    req.callback = [&](StatusCode code, std::vector<cv::Mat>, const std::string&) {
        called = true;
        resultCode = code;
    };

    auto s = server.InferAsync(std::move(req));
    EXPECT_TRUE(s);

    // 等待 batcher 线程处理
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_TRUE(called);
    EXPECT_EQ(resultCode, StatusCode::ErrorInternal);
}

// 测试：ReplaceModel 加载新版本模型
TEST_F(InferenceServerTest, ReplaceModelUpdatesVersion) {
    auto& server = InferenceServer::Instance();
    server.LoadModel("replace_me", "v1.onnx", "onnxruntime", 256, 1);
    EXPECT_TRUE(server.IsLoaded("replace_me"));

    auto s = server.ReplaceModel("replace_me", "v2.onnx", "onnxruntime", 512, 2);
    EXPECT_TRUE(s);
}

// 测试：SetBatchConfig 设置不崩溃
TEST_F(InferenceServerTest, SetBatchConfig) {
    auto& server = InferenceServer::Instance();
    server.SetBatchConfig(64, 10);
    // 无返回值，验证不崩溃即可
}

// 测试：Shutdown 清空所有队列和模型
TEST_F(InferenceServerTest, ShutdownCleansUp) {
    auto& server = InferenceServer::Instance();
    server.LoadModel("cleanup_test", "dummy.onnx", "onnxruntime", 256, 1);
    EXPECT_TRUE(server.IsLoaded("cleanup_test"));

    server.Shutdown();
    // 重启后应能正常加载新模型
    server.LoadModel("cleanup_test", "dummy.onnx", "onnxruntime", 256, 1);
    EXPECT_TRUE(server.IsLoaded("cleanup_test"));
}

// 测试：多次 Shutdown 不崩溃
TEST_F(InferenceServerTest, DoubleShutdownNoCrash) {
    auto& server = InferenceServer::Instance();
    server.LoadModel("double_shutdown", "dummy.onnx", "onnxruntime", 256, 1);
    server.Shutdown();
    server.Shutdown(); // 第二次不应崩溃
}

// 测试：加载多种后端类型不崩溃
TEST_F(InferenceServerTest, LoadMultipleBackendTypes) {
    auto& server = InferenceServer::Instance();
    auto s1 = server.LoadModel("model_trt", "dummy.engine", "tensorrt", 512, 1);
    EXPECT_TRUE(s1);

    auto s2 = server.LoadModel("model_ort", "dummy.onnx", "onnxruntime", 256, 1);
    EXPECT_TRUE(s2);

    auto s3 = server.LoadModel("model_lt", "dummy.pt", "libtorch", 1024, 1);
    EXPECT_TRUE(s3);

    std::string models = server.ListModels();
    EXPECT_NE(models.find("model_trt"), std::string::npos);
    EXPECT_NE(models.find("model_ort"), std::string::npos);
    EXPECT_NE(models.find("model_lt"), std::string::npos);
}
