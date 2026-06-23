#include <gtest/gtest.h>
#include "api/aicore_api.h"
#include "core/types.h"
#include <string>

using namespace aicore;

TEST(AICoreApiTest, Version) {
    const char* ver = aicore_version();
    ASSERT_NE(ver, nullptr);
    EXPECT_EQ(std::string(ver), "0.1.0");
}

TEST(AICoreApiTest, PipelineCreateNull) {
    const char* err = nullptr;
    AICorePipeline p = aicore_pipeline_create(nullptr, &err);
    EXPECT_EQ(p, nullptr);
    ASSERT_NE(err, nullptr);
    EXPECT_NE(std::string(err), "");
}

TEST(AICoreApiTest, PipelineCreateInvalidConfig) {
    const char* err = nullptr;
    AICorePipeline p = aicore_pipeline_create("not valid json [[[", &err);
    EXPECT_EQ(p, nullptr);
    ASSERT_NE(err, nullptr);
    EXPECT_NE(std::string(err), "");
}

TEST(AICoreApiTest, PipelineCreateNullErrorOut) {
    AICorePipeline p = aicore_pipeline_create(nullptr, nullptr);
    EXPECT_EQ(p, nullptr);
}

TEST(AICoreApiTest, PipelineExecuteNull) {
    unsigned char dummy[4] = {0};
    AICoreResult result = nullptr;
    const char* err = nullptr;

    int ret = aicore_pipeline_execute(nullptr, dummy, 2, 2, 1, &result, &err);
    EXPECT_EQ(ret, -1);
    ASSERT_NE(err, nullptr);
    EXPECT_NE(std::string(err), "");
}

TEST(AICoreApiTest, PipelineExecuteInvalidParams) {
    AICorePipeline dummy = reinterpret_cast<AICorePipeline>(static_cast<uintptr_t>(0x1));
    AICoreResult result = nullptr;
    const char* err = nullptr;

    int ret = aicore_pipeline_execute(dummy, nullptr, 10, 10, 3, &result, &err);
    EXPECT_EQ(ret, -1);

    unsigned char data[4] = {0};
    ret = aicore_pipeline_execute(dummy, data, 0, 10, 3, &result, &err);
    EXPECT_EQ(ret, -1);

    ret = aicore_pipeline_execute(dummy, data, 10, 0, 3, &result, &err);
    EXPECT_EQ(ret, -1);
}

TEST(AICoreApiTest, ResultToJsonNull) {
    const char* json = aicore_result_to_json(nullptr);
    ASSERT_NE(json, nullptr);
    EXPECT_EQ(std::string(json), "{}");
}

TEST(AICoreApiTest, ResultToJsonEmptyResult) {
    auto* r = new Result();
    r->timestamp = 0;
    r->totalLatencyMs = 0;
    r->status = StatusCode::OK;

    AICoreResult handle = static_cast<AICoreResult>(r);
    const char* json = aicore_result_to_json(handle);
    ASSERT_NE(json, nullptr);
    std::string s(json);

    EXPECT_NE(s.find("\"detections\":["), std::string::npos);
    EXPECT_NE(s.find("\"status\":0"), std::string::npos);

    aicore_result_free(handle);
}

TEST(AICoreApiTest, ResultToJsonWithDetections) {
    auto* r = new Result();
    r->timestamp = 12345678;
    r->totalLatencyMs = 42.5;
    r->status = StatusCode::OK;

    NodeResult det;
    det.nodeId = "test_node";
    det.label = "defect";
    det.confidence = 1.0f;
    det.bbox = {0, 0, 10, 10};
    r->detections.push_back(det);

    AICoreResult handle = static_cast<AICoreResult>(r);
    const char* json = aicore_result_to_json(handle);
    ASSERT_NE(json, nullptr);
    std::string s(json);

    EXPECT_NE(s.find("\"timestamp\":12345678"), std::string::npos);
    EXPECT_NE(s.find("\"node_id\":\"test_node\""), std::string::npos);
    EXPECT_NE(s.find("\"label\":\"defect\""), std::string::npos);
    EXPECT_NE(s.find("\"x\":0"), std::string::npos);
    EXPECT_NE(s.find("\"w\":10"), std::string::npos);

    aicore_result_free(handle);
}

TEST(AICoreApiTest, ResultToJsonWithMeasurements) {
    auto* r = new Result();
    r->timestamp = 1;
    r->totalLatencyMs = 5.0;
    r->status = StatusCode::OK;

    NodeResult det;
    det.nodeId = "m_node";
    det.label = "scratch";
    det.confidence = 1.0f;
    det.bbox = {0, 0, 10, 10};
    det.measurements["length"] = 15.0;
    det.measurements["width"] = 3.0;
    r->detections.push_back(det);

    AICoreResult handle = static_cast<AICoreResult>(r);
    const char* json = aicore_result_to_json(handle);
    ASSERT_NE(json, nullptr);
    std::string s(json);

    EXPECT_NE(s.find("\"length\":15"), std::string::npos);
    EXPECT_NE(s.find("\"width\":3"), std::string::npos);
    EXPECT_EQ(s.find("\"anomaly_score\":"), std::string::npos);

    aicore_result_free(handle);
}

TEST(AICoreApiTest, ResultToJsonWithAnomalyScore) {
    auto* r = new Result();
    r->timestamp = 1;
    r->totalLatencyMs = 5.0;
    r->status = StatusCode::OK;

    NodeResult det;
    det.nodeId = "a_node";
    det.label = "anomaly";
    det.confidence = 1.0f;
    det.bbox = {0, 0, 10, 10};
    det.measurements["anomaly_score"] = 0.87;
    r->detections.push_back(det);

    AICoreResult handle = static_cast<AICoreResult>(r);
    const char* json = aicore_result_to_json(handle);
    ASSERT_NE(json, nullptr);
    std::string s(json);

    EXPECT_NE(s.find("anomaly_score"), std::string::npos);

    aicore_result_free(handle);
}

TEST(AICoreApiTest, ResultToJsonMultipleDetections) {
    auto* r = new Result();
    r->timestamp = 0;
    r->totalLatencyMs = 0;
    r->status = StatusCode::OK;

    for (int i = 0; i < 3; i++) {
        NodeResult det;
        det.nodeId = "n" + std::to_string(i);
        det.label = "obj";
        det.confidence = 1.0f;
        det.bbox = {0, 0, 10, 10};
        r->detections.push_back(det);
    }

    AICoreResult handle = static_cast<AICoreResult>(r);
    const char* json = aicore_result_to_json(handle);
    ASSERT_NE(json, nullptr);
    std::string s(json);

    EXPECT_NE(s.find("\"node_id\":\"n0\""), std::string::npos);
    EXPECT_NE(s.find("\"node_id\":\"n2\""), std::string::npos);

    aicore_result_free(handle);
}

TEST(AICoreApiTest, ResultFreeNull) {
    aicore_result_free(nullptr);
}

TEST(AICoreApiTest, PipelineDestroyNull) {
    aicore_pipeline_destroy(nullptr);
}

TEST(AICoreApiTest, PipelineGetStateNull) {
    int state = aicore_pipeline_get_state(nullptr);
    EXPECT_EQ(state, -1);
}

TEST(AICoreApiTest, EngineInitNull) {
    const char* err = nullptr;
    int ret = aicore_engine_init(nullptr, &err);
    EXPECT_EQ(ret, -1);
    ASSERT_NE(err, nullptr);
    EXPECT_NE(std::string(err), "");
}

TEST(AICoreApiTest, EngineInitNullErrorOut) {
    int ret = aicore_engine_init(nullptr, nullptr);
    EXPECT_EQ(ret, -1);
}

TEST(AICoreApiTest, EngineExecuteNull) {
    AICoreResult result = nullptr;
    int ret = aicore_engine_execute(nullptr, 10, 10, 3, &result, nullptr);
    EXPECT_EQ(ret, -1);
}
