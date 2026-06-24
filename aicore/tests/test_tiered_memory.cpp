// ============================================================
// 文件: tests/test_tiered_memory.cpp
// 用途: TieredMemoryBank + MemoryManager 单元测试
//   涵盖内存银行分层管理、LRU 驱逐策略
//   所有测试均不依赖 CUDA GPU（仅测试 CPU/Disk tier 逻辑和边界条件）
// ============================================================

#include <gtest/gtest.h>
#include "patchcore/tiered_memory_bank.h"
#include "patchcore/memory_manager.h"
#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace aicore;

namespace {

// 创建有效的 memory bank 测试文件
// 二进制格式: magic(4B) + num(4B) + dim(4B) + float[num * (dim + 3)]
void CreateTestBankFile(const std::string& path, int num, int dim) {
    std::ofstream ofs(path, std::ios::binary);
    ASSERT_TRUE(ofs);

    uint32_t magic = TieredMemoryBank::kMagic;
    uint32_t numU32 = static_cast<uint32_t>(num);
    uint32_t dimU32 = static_cast<uint32_t>(dim);

    ofs.write(reinterpret_cast<const char*>(&magic), 4);
    ofs.write(reinterpret_cast<const char*>(&numU32), 4);
    ofs.write(reinterpret_cast<const char*>(&dimU32), 4);

    int stride = dim + 3;
    for (int i = 0; i < num; i++) {
        std::vector<float> row(stride, 0.0f);
        for (int j = 0; j < dim; j++) {
            row[j] = static_cast<float>(i * dim + j);
        }
        ofs.write(reinterpret_cast<const char*>(row.data()),
                  stride * sizeof(float));
    }
}

// 小文件 — 单特征全零，用于精确匹配测试
void CreateZerosBankFile(const std::string& path, int dim) {
    std::ofstream ofs(path, std::ios::binary);
    ASSERT_TRUE(ofs);

    uint32_t magic = TieredMemoryBank::kMagic;
    uint32_t num = 1;
    uint32_t dimU32 = static_cast<uint32_t>(dim);

    ofs.write(reinterpret_cast<const char*>(&magic), 4);
    ofs.write(reinterpret_cast<const char*>(&num), 4);
    ofs.write(reinterpret_cast<const char*>(&dimU32), 4);

    int stride = dim + 3;
    std::vector<float> row(stride, 0.0f);
    ofs.write(reinterpret_cast<const char*>(row.data()),
              stride * sizeof(float));
}

std::string testPath(const char* name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

void removeIfPossible(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

} // anonymous namespace

// ─── TieredMemoryBank ───────────────────────────────────────

TEST(TieredMemoryBankTest, DefaultState) {
    TieredMemoryBank bank;
    EXPECT_EQ(bank.Size(), 0);
    EXPECT_EQ(bank.FeatureDim(), 0);
    EXPECT_EQ(bank.GetTier(), BankTier::kDisk);
}

TEST(TieredMemoryBankTest, MoveConstructor) {
    CreateTestBankFile(testPath("test_move_src.bin"), 3, 4);

    TieredMemoryBank src;
    ASSERT_TRUE(src.Load(testPath("test_move_src.bin")));

    TieredMemoryBank dst(std::move(src));
    EXPECT_EQ(dst.Size(), 3);
    EXPECT_EQ(dst.FeatureDim(), 4);
    EXPECT_EQ(dst.GetTier(), BankTier::kDisk);
    EXPECT_EQ(src.Size(), 0);
    EXPECT_EQ(src.FeatureDim(), 0);

    // 源对象再次 clear 不崩溃
    src.Clear();
    removeIfPossible(testPath("test_move_src.bin"));
}

TEST(TieredMemoryBankTest, MoveAssignment) {
    CreateTestBankFile(testPath("test_move_assign.bin"), 5, 2);

    TieredMemoryBank src;
    ASSERT_TRUE(src.Load(testPath("test_move_assign.bin")));

    TieredMemoryBank dst;
    dst = std::move(src);
    EXPECT_EQ(dst.Size(), 5);
    EXPECT_EQ(dst.FeatureDim(), 2);
    EXPECT_EQ(src.Size(), 0);

    removeIfPossible(testPath("test_move_assign.bin"));
}

TEST(TieredMemoryBankTest, ClearEmpty) {
    TieredMemoryBank bank;
    bank.Clear();  // 不得崩溃
    EXPECT_EQ(bank.Size(), 0);
}

TEST(TieredMemoryBankTest, DoubleClear) {
    CreateTestBankFile(testPath("test_double_clear.bin"), 2, 3);

    TieredMemoryBank bank;
    ASSERT_TRUE(bank.Load(testPath("test_double_clear.bin")));
    bank.Clear();
    bank.Clear();  // 第二次 clear 不得崩溃
    EXPECT_EQ(bank.Size(), 0);

    removeIfPossible(testPath("test_double_clear.bin"));
}

TEST(TieredMemoryBankTest, LoadNonExistent) {
    TieredMemoryBank bank;
    auto s = bank.Load(testPath("nonexistent_file_xyz.bin"));
    EXPECT_FALSE(s);
}

TEST(TieredMemoryBankTest, LoadInvalidMagic) {
    {
        std::ofstream ofs(testPath("test_bad_magic.bin"), std::ios::binary);
        ASSERT_TRUE(ofs);
        uint32_t badMagic = 0xDEADBEEF;
        ofs.write(reinterpret_cast<const char*>(&badMagic), 4);
    }

    TieredMemoryBank bank;
    auto s = bank.Load(testPath("test_bad_magic.bin"));
    EXPECT_FALSE(s);

    removeIfPossible(testPath("test_bad_magic.bin"));
}

TEST(TieredMemoryBankTest, LoadEmptyFile) {
    {
        std::ofstream ofs(testPath("test_empty.bin"), std::ios::binary);
        ASSERT_TRUE(ofs);
        // 只写入 magic，没有 num/dim
        uint32_t magic = TieredMemoryBank::kMagic;
        ofs.write(reinterpret_cast<const char*>(&magic), 4);
    }

    TieredMemoryBank bank;
    // 文件过短，mmap 内存中的 num/dim 读取 undefined — 但不会崩溃
    // 这个测试验证极端边界不崩溃
    auto s = bank.Load(testPath("test_empty.bin"));
    // 结果无所谓（未定义数据），关键是不得崩溃
    (void)s;
    bank.Clear();
    removeIfPossible(testPath("test_empty.bin"));
}

TEST(TieredMemoryBankTest, LoadValidFile) {
    CreateTestBankFile(testPath("test_valid.bin"), 10, 4);

    TieredMemoryBank bank;
    auto s = bank.Load(testPath("test_valid.bin"));
    ASSERT_TRUE(s);
    EXPECT_EQ(bank.Size(), 10);
    EXPECT_EQ(bank.FeatureDim(), 4);
    EXPECT_EQ(bank.GetTier(), BankTier::kDisk);

    bank.Clear();
    removeIfPossible(testPath("test_valid.bin"));
}

TEST(TieredMemoryBankTest, ComputeAnomalyMapEmptyQueries) {
    CreateTestBankFile(testPath("test_empty_query.bin"), 5, 3);

    TieredMemoryBank bank;
    ASSERT_TRUE(bank.Load(testPath("test_empty_query.bin")));

    std::vector<PatchFeature> emptyQueries;
    auto result = bank.ComputeAnomalyMap(emptyQueries, 100, 100);
    EXPECT_TRUE(result.empty());

    bank.Clear();
    removeIfPossible(testPath("test_empty_query.bin"));
}

TEST(TieredMemoryBankTest, ComputeAnomalyMapEmptyBank) {
    TieredMemoryBank bank;

    std::vector<PatchFeature> queries;
    PatchFeature pf;
    pf.features = {1.0f, 2.0f};
    pf.patchRow = 0;
    pf.patchCol = 0;
    queries.push_back(pf);

    auto result = bank.ComputeAnomalyMap(queries, 100, 100);
    EXPECT_TRUE(result.empty());
}

TEST(TieredMemoryBankTest, ComputeOnCPUCorrectShape) {
    CreateTestBankFile(testPath("test_cpu_shape.bin"), 10, 4);

    TieredMemoryBank bank;
    ASSERT_TRUE(bank.Load(testPath("test_cpu_shape.bin")));

    // 2 个查询 patch
    std::vector<PatchFeature> queries;
    for (int i = 0; i < 2; i++) {
        PatchFeature pf;
        pf.features = {0.0f, 0.0f, 0.0f, 0.0f};
        pf.patchRow = i;
        pf.patchCol = 0;
        queries.push_back(pf);
    }

    auto result = bank.ComputeAnomalyMap(queries, 32, 32);
    // 结果是被上采样到 32x32 的得分图
    EXPECT_EQ(result.size(), static_cast<size_t>(32 * 32));

    bank.Clear();
    removeIfPossible(testPath("test_cpu_shape.bin"));
}

TEST(TieredMemoryBankTest, ComputeOnCPUMultiLayer) {
    CreateTestBankFile(testPath("test_multi_layer.bin"), 10, 4);

    TieredMemoryBank bank;
    ASSERT_TRUE(bank.Load(testPath("test_multi_layer.bin")));

    std::vector<PatchFeature> queries;
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            PatchFeature q;
            q.features = {0, 0, 0, 0};
            q.layerIdx = 0; q.patchRow = r; q.patchCol = c;
            queries.push_back(q);
        }
    }
    PatchFeature ql;
    ql.features = {0, 0, 0, 0};
    ql.layerIdx = 1; ql.patchRow = 0; ql.patchCol = 0;
    queries.push_back(ql);

    auto result = bank.ComputeAnomalyMap(queries, 8, 8);
    EXPECT_EQ(result.size(), static_cast<size_t>(64));

    bank.Clear();
    removeIfPossible(testPath("test_multi_layer.bin"));
}

TEST(TieredMemoryBankTest, ComputeOnCPUExactMatch) {
    CreateZerosBankFile(testPath("test_exact.bin"), 2);

    TieredMemoryBank bank;
    ASSERT_TRUE(bank.Load(testPath("test_exact.bin")));

    PatchFeature pf;
    pf.features = {0.0f, 0.0f};
    pf.patchRow = 0;
    pf.patchCol = 0;

    auto result = bank.ComputeAnomalyMap({pf}, 4, 4);
    ASSERT_EQ(result.size(), static_cast<size_t>(16));
    // 距离 0，所以得分图所有像素应接近 0
    for (float v : result) {
        EXPECT_NEAR(v, 0.0f, 1e-4f);
    }

    bank.Clear();
    removeIfPossible(testPath("test_exact.bin"));
}

TEST(TieredMemoryBankTest, PromoteToGPUNotLoaded) {
    TieredMemoryBank bank;
    auto s = bank.PromoteToGPU();
    EXPECT_FALSE(s);
}

TEST(TieredMemoryBankTest, DemoteToCPUFromDisk) {
    CreateTestBankFile(testPath("test_demote_cpu.bin"), 3, 2);

    TieredMemoryBank bank;
    ASSERT_TRUE(bank.Load(testPath("test_demote_cpu.bin")));
    EXPECT_EQ(bank.GetTier(), BankTier::kDisk);

    // disk tier 下调降级不应有影响
    bank.DemoteToCPU();
    EXPECT_EQ(bank.GetTier(), BankTier::kDisk);

    bank.Clear();
    removeIfPossible(testPath("test_demote_cpu.bin"));
}

TEST(TieredMemoryBankTest, ClearAfterLoadResets) {
    CreateTestBankFile(testPath("test_reset.bin"), 7, 8);

    TieredMemoryBank bank;
    ASSERT_TRUE(bank.Load(testPath("test_reset.bin")));
    EXPECT_EQ(bank.Size(), 7);

    bank.Clear();
    EXPECT_EQ(bank.Size(), 0);
    EXPECT_EQ(bank.FeatureDim(), 0);
    EXPECT_EQ(bank.GetTier(), BankTier::kDisk);

    removeIfPossible(testPath("test_reset.bin"));
}

TEST(TieredMemoryBankTest, LoadOverwritesPrevious) {
    CreateTestBankFile(testPath("test_first.bin"), 3, 2);
    CreateTestBankFile(testPath("test_second.bin"), 5, 4);

    TieredMemoryBank bank;
    ASSERT_TRUE(bank.Load(testPath("test_first.bin")));
    EXPECT_EQ(bank.Size(), 3);
    EXPECT_EQ(bank.FeatureDim(), 2);

    ASSERT_TRUE(bank.Load(testPath("test_second.bin")));
    EXPECT_EQ(bank.Size(), 5);
    EXPECT_EQ(bank.FeatureDim(), 4);

    bank.Clear();
    removeIfPossible(testPath("test_first.bin"));
    removeIfPossible(testPath("test_second.bin"));
}

// ─── MemoryManager ──────────────────────────────────────────

// 注意: MemoryManager 是进程级单例，测试间状态共享。
// 修改 budget 的测试需自行恢复。

TEST(MemoryManagerTest, GetInstanceSingleton) {
    auto& m1 = MemoryManager::GetInstance();
    auto& m2 = MemoryManager::GetInstance();
    EXPECT_EQ(&m1, &m2);
}

TEST(MemoryManagerTest, DefaultBudget) {
    auto& mgr = MemoryManager::GetInstance();
    EXPECT_GT(mgr.Budget(), 0);
}

TEST(MemoryManagerTest, AvailableDefaultsToBudget) {
    auto& mgr = MemoryManager::GetInstance();
    EXPECT_EQ(mgr.Available(), mgr.Budget());
}

TEST(MemoryManagerTest, SetBudgetAndBudget) {
    auto& mgr = MemoryManager::GetInstance();
    size_t oldBudget = mgr.Budget();
    mgr.SetBudget(999);
    EXPECT_EQ(mgr.Budget(), 999);
    mgr.SetBudget(oldBudget);
    EXPECT_EQ(mgr.Budget(), oldBudget);
}

TEST(MemoryManagerTest, TryAllocBudgetZero) {
    auto& mgr = MemoryManager::GetInstance();
    size_t oldBudget = mgr.Budget();

    mgr.SetBudget(0);
    uint64_t id = 999;
    float* ptr = mgr.TryAlloc(100, id);
    EXPECT_EQ(ptr, nullptr);

    mgr.SetBudget(oldBudget);
}

TEST(MemoryManagerTest, FreeUnknownId) {
    auto& mgr = MemoryManager::GetInstance();
    mgr.Free(999);  // 不得崩溃
    mgr.Free(0);
}

TEST(MemoryManagerTest, TouchUnknownId) {
    auto& mgr = MemoryManager::GetInstance();
    mgr.Touch(999);  // 不得崩溃
    mgr.Touch(0);
}
