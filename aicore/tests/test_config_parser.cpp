// ============================================================
// 文件: tests/test_config_parser.cpp
// 用途: 流水线配置解析器 (ConfigParser) 解析/序列化/错误处理测试
// ============================================================

#include <gtest/gtest.h>
#include "config/config_parser.h"

using namespace aicore;

// 有效的流水线 JSON 测试数据（resize + tensorrt 模型）
const char* kValidJson = R"({
    "pipeline": {
        "name": "test_pipeline",
        "max_concurrency": 2,
        "enable_profiling": true,
        "nodes": [
            {
                "id": "pre_1",
                "type": "resize",
                "params": {"width": "640", "height": "640"}
            },
            {
                "id": "model_1",
                "type": "model",
                "backend": "tensorrt",
                "model_path": "models/test.engine",
                "device_id": 0,
                "batch_size": 4
            }
        ],
        "edges": [
            {"from": "input", "to": "pre_1"},
            {"from": "pre_1", "to": "model_1"}
        ]
    }
})";

// 测试：解析有效 JSON 得到正确的流水线配置字段
TEST(ConfigParserTest, ParseValidJson) {
    ConfigParser parser;
    PipelineConfig config;
    Status s = parser.Parse(kValidJson, config);
    ASSERT_TRUE(s) << parser.GetLastError();

    EXPECT_EQ(config.name, "test_pipeline");
    EXPECT_EQ(config.maxConcurrency, 2);
    EXPECT_TRUE(config.enableProfiling);
    ASSERT_EQ(config.nodes.size(), 2);
    ASSERT_EQ(config.edges.size(), 2);
}

// 测试：解析出的节点字段值正确
TEST(ConfigParserTest, ParseNodes) {
    ConfigParser parser;
    PipelineConfig config;
    ASSERT_TRUE(parser.Parse(kValidJson, config));

    EXPECT_EQ(config.nodes[0].id, "pre_1");
    EXPECT_EQ(config.nodes[0].type, "resize");
    EXPECT_EQ(config.nodes[0].params.at("width"), "640");

    EXPECT_EQ(config.nodes[1].id, "model_1");
    EXPECT_EQ(config.nodes[1].backend, BackendType::kTensorRT);
    EXPECT_EQ(config.nodes[1].modelPath, "models/test.engine");
    EXPECT_EQ(config.nodes[1].deviceId, 0);
    EXPECT_EQ(config.nodes[1].batchSize, 4);
}

// 测试：解析出的边连接关系正确
TEST(ConfigParserTest, ParseEdges) {
    ConfigParser parser;
    PipelineConfig config;
    ASSERT_TRUE(parser.Parse(kValidJson, config));

    EXPECT_EQ(config.edges[0].from, "input");
    EXPECT_EQ(config.edges[0].to, "pre_1");
    EXPECT_EQ(config.edges[1].from, "pre_1");
    EXPECT_EQ(config.edges[1].to, "model_1");
}

// 测试：解析非法 JSON 返回失败并有错误信息
TEST(ConfigParserTest, InvalidJson) {
    ConfigParser parser;
    PipelineConfig config;
    Status s = parser.Parse("{invalid", config);
    EXPECT_FALSE(s);
    EXPECT_FALSE(parser.GetLastError().empty());
}

// 测试：序列化 → 重新解析后数据一致（往返测试）
TEST(ConfigParserTest, SerializeRoundtrip) {
    ConfigParser parser;
    PipelineConfig config1;
    ASSERT_TRUE(parser.Parse(kValidJson, config1));

    std::string serialized;
    ASSERT_TRUE(parser.Serialize(config1, serialized));

    PipelineConfig config2;
    ASSERT_TRUE(parser.Parse(serialized, config2));

    EXPECT_EQ(config2.name, config1.name);
    EXPECT_EQ(config2.nodes.size(), config1.nodes.size());
    EXPECT_EQ(config2.edges.size(), config1.edges.size());
    EXPECT_EQ(config2.nodes[1].backend, config1.nodes[1].backend);
}
