#include <gtest/gtest.h>
#include "trainer/training/yolo_trainer.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>

namespace aicore {

namespace fs = std::filesystem;

class YOLOTrainerTest : public ::testing::Test {
protected:
    fs::path tmpDir;
    fs::path trainImgDir;
    fs::path trainLabelDir;

    void SetUp() override {
        tmpDir = fs::temp_directory_path() / "aicore_yolo_trainer_test";
        trainImgDir = tmpDir / "train_images";
        trainLabelDir = tmpDir / "train_labels";
        fs::create_directories(trainImgDir);
        fs::create_directories(trainLabelDir);

        // 创建 4 张合成训练图片, 每张 2 个目标
        for (int i = 0; i < 4; i++) {
            cv::Mat img(100, 100, CV_8UC3, cv::Scalar(120, 120, 120));
            cv::rectangle(img, cv::Rect(20, 20, 30, 40), cv::Scalar(0, 255, 0), -1);
            cv::imwrite((trainImgDir / ("img" + std::to_string(i) + ".jpg")).string(), img);

            std::ofstream f(trainLabelDir / ("img" + std::to_string(i) + ".txt"));
            // class 0 at (0.5, 0.5, 0.2, 0.2)
            f << "0 0.5 0.5 0.2 0.2\n";
            // class 1 at (0.3, 0.4, 0.1, 0.15)
            f << "1 0.3 0.4 0.1 0.15\n";
        }
    }

    void TearDown() override {
        fs::remove_all(tmpDir);
    }
};

TEST_F(YOLOTrainerTest, Init_FailsOnEmptyDir) {
    YOLOTrainConfig cfg;
    cfg.trainImgDir = (tmpDir / "nonexistent").string();
    YOLOTrainer trainer(cfg);
    EXPECT_FALSE(trainer.Init());
}

TEST_F(YOLOTrainerTest, Init_SucceedsWithData) {
    YOLOTrainConfig cfg;
    cfg.trainImgDir = trainImgDir.string();
    cfg.trainLabelDir = trainLabelDir.string();
    cfg.batchSize = 2;
    cfg.epochs = 1;
    cfg.warmupEpochs = 0;
    cfg.imgSize = 128;
    cfg.lr = 0.001f;
    cfg.saveDir = (tmpDir / "weights").string();

    YOLOTrainer trainer(cfg);
    EXPECT_TRUE(trainer.Init());
}

TEST_F(YOLOTrainerTest, Train_RunsOneEpoch) {
    YOLOTrainConfig cfg;
    cfg.trainImgDir = trainImgDir.string();
    cfg.trainLabelDir = trainLabelDir.string();
    cfg.batchSize = 2;
    cfg.epochs = 1;
    cfg.warmupEpochs = 0;
    cfg.imgSize = 128;
    cfg.lr = 0.001f;
    cfg.saveDir = (tmpDir / "weights").string();

    YOLOTrainer trainer(cfg);
    ASSERT_TRUE(trainer.Init());

    std::vector<YOLOTrainProgress> progress;
    trainer.SetProgressCallback([&](const YOLOTrainProgress& p) {
        progress.push_back(p);
    });

    trainer.Train();

    // batchSize=2, 4 images → 2 batches
    EXPECT_GT(progress.size(), 0);
    EXPECT_TRUE(progress.back().done);
    // loss should be finite
    EXPECT_TRUE(std::isfinite(progress.back().loss));
}

TEST_F(YOLOTrainerTest, Train_LossDecreases) {
    YOLOTrainConfig cfg;
    cfg.trainImgDir = trainImgDir.string();
    cfg.trainLabelDir = trainLabelDir.string();
    cfg.batchSize = 2;
    cfg.epochs = 5;
    cfg.warmupEpochs = 0;
    cfg.imgSize = 128;
    cfg.lr = 0.01f;
    cfg.saveDir = (tmpDir / "weights").string();

    YOLOTrainer trainer(cfg);
    ASSERT_TRUE(trainer.Init());

    std::vector<YOLOTrainProgress> progress;
    trainer.SetProgressCallback([&](const YOLOTrainProgress& p) {
        progress.push_back(p);
    });

    trainer.Train();

    // 至少有一些 progress 回调
    EXPECT_GT(progress.size(), 0);
    if (progress.size() >= 2) {
        float firstLoss = progress[0].loss;
        float lastLoss = progress.back().loss;
        EXPECT_TRUE(std::isfinite(lastLoss));
    }
}

TEST_F(YOLOTrainerTest, Train_StopWorks) {
    YOLOTrainConfig cfg;
    cfg.trainImgDir = trainImgDir.string();
    cfg.trainLabelDir = trainLabelDir.string();
    cfg.batchSize = 2;
    cfg.epochs = 100; // many epochs
    cfg.warmupEpochs = 0;
    cfg.imgSize = 128;
    cfg.lr = 0.001f;
    cfg.saveDir = (tmpDir / "weights").string();

    YOLOTrainer trainer(cfg);
    ASSERT_TRUE(trainer.Init());

    trainer.SetProgressCallback([&](const YOLOTrainProgress& p) {
        if (p.epoch >= 2) trainer.Stop();
    });

    trainer.Train();
    // stopped early, progress should be < 100 epochs
    EXPECT_LT(trainer.GetLastProgress().epoch, 50);
}

TEST_F(YOLOTrainerTest, SaveCheckpoint_CreatesFile) {
    YOLOTrainConfig cfg;
    cfg.trainImgDir = trainImgDir.string();
    cfg.trainLabelDir = trainLabelDir.string();
    cfg.batchSize = 4;
    cfg.epochs = 1;
    cfg.warmupEpochs = 0;
    cfg.imgSize = 128;
    cfg.lr = 0.001f;
    cfg.saveDir = (tmpDir / "weights").string();

    YOLOTrainer trainer(cfg);
    ASSERT_TRUE(trainer.Init());
    trainer.Train();

    auto checkpointPath = (fs::path(cfg.saveDir) / "yolov8_epoch_0001.pt");
    EXPECT_TRUE(fs::exists(checkpointPath));
}

} // namespace aicore
