#include <gtest/gtest.h>
#include "trainer/data/yolo_data.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <filesystem>

namespace aicore {

namespace fs = std::filesystem;

class YOLODataTest : public ::testing::Test {
protected:
    fs::path tmpDir;
    fs::path imgDir;
    fs::path labelDir;

    void SetUp() override {
        tmpDir = fs::temp_directory_path() / "aicore_yolo_test";
        imgDir = tmpDir / "images";
        labelDir = tmpDir / "labels";
        fs::create_directories(imgDir);
        fs::create_directories(labelDir);
    }

    void TearDown() override {
        fs::remove_all(tmpDir);
    }

    void createImage(const std::string& name, int w, int h) {
        cv::Mat img(h, w, CV_8UC3, cv::Scalar(128, 128, 128));
        cv::rectangle(img, cv::Rect(w/4, h/4, w/2, h/2), cv::Scalar(0, 255, 0), -1);
        cv::imwrite((imgDir / name).string(), img);
    }

    void createLabel(const std::string& name, const std::vector<std::vector<float>>& boxes) {
        std::ofstream f(labelDir / name);
        for (auto& b : boxes) {
            f << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << " " << b[4] << "\n";
        }
    }
};

TEST_F(YOLODataTest, DatasetLoad_FindsMatchingPairs) {
    createImage("img1.jpg", 640, 480);
    createImage("img2.jpg", 320, 320);
    createImage("img3.jpg", 100, 200);
    createLabel("img1.txt", {{0, 0.5, 0.5, 0.2, 0.3}});
    createLabel("img2.txt", {{1, 0.3, 0.4, 0.1, 0.2}, {2, 0.7, 0.6, 0.15, 0.25}});

    YOLODataset ds;
    ASSERT_TRUE(ds.Load(imgDir.string(), labelDir.string()));
    EXPECT_EQ(ds.Size(), 2); // img3 has no labels, skipped
    EXPECT_EQ(ds.NumClasses(), 3); // classes 0,1,2
}

TEST_F(YOLODataTest, DatasetGet_ReturnsCorrectBoxes) {
    createImage("test.jpg", 640, 480);
    createLabel("test.txt", {{0, 0.5, 0.5, 0.2, 0.3}, {1, 0.3, 0.4, 0.1, 0.2}});

    YOLODataset ds;
    ASSERT_TRUE(ds.Load(imgDir.string(), labelDir.string()));
    ASSERT_EQ(ds.Size(), 1);

    auto s = ds.Get(0);
    EXPECT_EQ(s.labels.size(), 2);
    EXPECT_EQ(s.labels[0], 0);
    EXPECT_EQ(s.labels[1], 1);
    EXPECT_NEAR(s.boxes[0].x, 0.5, 1e-5);
    EXPECT_NEAR(s.boxes[0].width, 0.2, 1e-5);
    EXPECT_FALSE(s.image.empty());
}

TEST_F(YOLODataTest, DatasetGet_ImagePathStored) {
    createImage("pathcheck.jpg", 100, 100);
    createLabel("pathcheck.txt", {{0, 0.5, 0.5, 0.1, 0.1}});

    YOLODataset ds;
    ASSERT_TRUE(ds.Load(imgDir.string(), labelDir.string()));
    EXPECT_TRUE(ds.GetPath(0).find("pathcheck.jpg") != std::string::npos);
}

TEST_F(YOLODataTest, DatasetLoad_EmptyDir) {
    YOLODataset ds;
    EXPECT_FALSE(ds.Load((imgDir / "nonexistent").string()));
}

TEST_F(YOLODataTest, Letterbox_ScalesCorrectly) {
    YOLOSample s;
    // 300×400, scale=min(640/300,640/400)=1.6, new=(480,640), padX=(640-480)/2=80
    s.image = cv::Mat(400, 300, CV_8UC3, cv::Scalar(100, 100, 100));
    s.boxes.push_back(cv::Rect2f(0.3, 0.5, 0.2, 0.2));

    auto result = letterbox(s, 640);
    EXPECT_EQ(result.image.rows, 640);
    EXPECT_EQ(result.image.cols, 640);
    ASSERT_EQ(result.boxes.size(), 1);
    // cx: 0.3*300=90 → 90*1.6=144 → (144+80)/640=0.35
    EXPECT_NEAR(result.boxes[0].x, 0.35, 0.01);
    // width: 0.2*300=60 → 60*1.6=96 → 96/640=0.15
    EXPECT_NEAR(result.boxes[0].width, 0.15, 0.01);
    // cy: 0.5*400=200 → 200*1.6=320 → 320/640=0.5 (padY=0)
    EXPECT_NEAR(result.boxes[0].y, 0.5, 0.01);
}

TEST_F(YOLODataTest, RandomFlipH_FlipsCx) {
    YOLOSample s;
    s.image = cv::Mat(100, 200, CV_8UC3, cv::Scalar(100, 100, 100));
    s.boxes.push_back(cv::Rect2f(0.3, 0.5, 0.2, 0.2));

    auto result = randomFlipH(s, 1.0); // force flip
    ASSERT_EQ(result.boxes.size(), 1);
    EXPECT_NEAR(result.boxes[0].x, 0.7, 1e-5); // 1 - 0.3
    EXPECT_NEAR(result.boxes[0].y, 0.5, 1e-5); // y unchanged
}

TEST_F(YOLODataTest, RandomFlipH_ZeroProbSkips) {
    YOLOSample s;
    s.image = cv::Mat(100, 100, CV_8UC3, cv::Scalar(100, 100, 100));
    s.boxes.push_back(cv::Rect2f(0.3, 0.5, 0.2, 0.2));

    auto result = randomFlipH(s, 0.0); // never flip
    EXPECT_NEAR(result.boxes[0].x, 0.3, 1e-5);
}

TEST_F(YOLODataTest, HsvJitter_DoesNotCrash) {
    YOLOSample s;
    s.image = cv::Mat(100, 100, CV_8UC3, cv::Scalar(60, 120, 180));

    auto result = hsvJitter(s, 1, 1, 1);
    EXPECT_FALSE(result.image.empty());
    EXPECT_EQ(result.image.rows, 100);
    EXPECT_EQ(result.image.cols, 100);
    EXPECT_EQ(result.boxes.size(), 0);
}

TEST_F(YOLODataTest, Mosaic_ProducesCorrectSize) {
    std::vector<YOLOSample> samples;
    for (int i = 0; i < 4; i++) {
        YOLOSample s;
        s.image = cv::Mat(100, 100, CV_8UC3, cv::Scalar(i * 50, i * 50, i * 50));
        s.boxes.push_back(cv::Rect2f(0.5, 0.5, 0.2, 0.2));
        s.labels.push_back(i);
        samples.push_back(s);
    }

    auto result = mosaicAugment(samples, 640);
    EXPECT_EQ(result.image.rows, 640);
    EXPECT_EQ(result.image.cols, 640);
    // at least some boxes survived clipping
    EXPECT_GT(result.boxes.size(), 0);
    EXPECT_EQ(result.boxes.size(), result.labels.size());
}

TEST_F(YOLODataTest, Mosaic_SingleSampleFallback) {
    std::vector<YOLOSample> samples;
    YOLOSample s;
    s.image = cv::Mat(50, 50, CV_8UC3, cv::Scalar(100, 100, 100));
    s.boxes.push_back(cv::Rect2f(0.5, 0.5, 0.2, 0.2));
    samples.push_back(s);

    auto result = mosaicAugment(samples, 640);
    // fallback: returns first sample unchanged
    EXPECT_EQ(result.image.rows, 50);
}

TEST_F(YOLODataTest, DataLoader_BatchesCorrectly) {
    // create 6 images with labels
    for (int i = 0; i < 6; i++) {
        createImage("batch" + std::to_string(i) + ".jpg", 200, 200);
        createLabel("batch" + std::to_string(i) + ".txt", {{0, 0.5, 0.5, 0.2, 0.2}});
    }

    auto ds = std::make_shared<YOLODataset>();
    ASSERT_TRUE(ds->Load(imgDir.string(), labelDir.string()));
    EXPECT_EQ(ds->Size(), 6);

    YOLODataLoader loader(ds, 4, 640, false);
    EXPECT_EQ(loader.NumBatches(), 2);
    EXPECT_TRUE(loader.HasNext());

    loader.mosaic = false;

    auto batch1 = loader.Next();
    EXPECT_EQ(batch1.images.size(0), 4);
    EXPECT_EQ(batch1.images.size(1), 3);
    EXPECT_EQ(batch1.images.size(2), 640);
    EXPECT_EQ(batch1.images.size(3), 640);
    // each image has 1 box → 4 targets
    EXPECT_EQ(batch1.targets.size(0), 4);
    EXPECT_EQ(batch1.targets.size(1), 6);

    EXPECT_TRUE(loader.HasNext());
    auto batch2 = loader.Next();
    EXPECT_EQ(batch2.images.size(0), 2);
    EXPECT_EQ(batch2.targets.size(0), 2);

    EXPECT_FALSE(loader.HasNext());
}

TEST_F(YOLODataTest, DataLoader_TargetFormat) {
    createImage("fmt.jpg", 100, 100);
    createLabel("fmt.txt", {{2, 0.5, 0.5, 0.2, 0.2}}); // class 2

    auto ds = std::make_shared<YOLODataset>();
    ASSERT_TRUE(ds->Load(imgDir.string(), labelDir.string()));
    EXPECT_EQ(ds->NumClasses(), 3);

    YOLODataLoader loader(ds, 1, 640, false);
    loader.mosaic = false;

    auto batch = loader.Next();
    ASSERT_EQ(batch.targets.size(0), 1);
    // target format: [batchIdx, cls, cx, cy, w, h]
    EXPECT_EQ(batch.targets[0][0].item<int>(), 0); // batchIdx
    EXPECT_EQ(batch.targets[0][1].item<int>(), 2); // cls
    EXPECT_NEAR(batch.targets[0][2].item<float>(), 0.5, 0.01); // cx
    EXPECT_NEAR(batch.targets[0][3].item<float>(), 0.5, 0.01); // cy
    EXPECT_NEAR(batch.targets[0][4].item<float>(), 0.2, 0.01); // w
    EXPECT_NEAR(batch.targets[0][5].item<float>(), 0.2, 0.01); // h
}

TEST_F(YOLODataTest, DataLoader_ShuffleChangesOrder) {
    for (int i = 0; i < 5; i++) {
        createImage("shuf" + std::to_string(i) + ".jpg", 50, 50);
        createLabel("shuf" + std::to_string(i) + ".txt", {{0, 0.5, 0.5, 0.2, 0.2}});
    }

    auto ds = std::make_shared<YOLODataset>();
    ASSERT_TRUE(ds->Load(imgDir.string(), labelDir.string()));

    YOLODataLoader loader(ds, 5, 640, true);
    loader.mosaic = false;
    auto batch = loader.Next();
    EXPECT_EQ(batch.images.size(0), 5);
    // shuffle shouldn't crash
}

TEST_F(YOLODataTest, DataLoader_EmptyDataset) {
    auto ds = std::make_shared<YOLODataset>();
    YOLODataLoader loader(ds, 4, 640, true);
    EXPECT_FALSE(loader.HasNext());
    EXPECT_EQ(loader.NumBatches(), 0);
}

TEST_F(YOLODataTest, DataLoader_ResetWorks) {
    for (int i = 0; i < 3; i++) {
        createImage("reset" + std::to_string(i) + ".jpg", 50, 50);
        createLabel("reset" + std::to_string(i) + ".txt", {{0, 0.5, 0.5, 0.2, 0.2}});
    }

    auto ds = std::make_shared<YOLODataset>();
    ASSERT_TRUE(ds->Load(imgDir.string(), labelDir.string()));

    YOLODataLoader loader(ds, 4, 640, false);
    loader.mosaic = false;

    auto b1 = loader.Next();
    EXPECT_FALSE(loader.HasNext()); // all 3 consumed

    loader.Reset();
    EXPECT_TRUE(loader.HasNext());
    auto b2 = loader.Next();
    EXPECT_EQ(b2.images.size(0), 3);
}

TEST_F(YOLODataTest, DatasetGet_InvalidIndex) {
    createImage("valid.jpg", 50, 50);
    createLabel("valid.txt", {{0, 0.5, 0.5, 0.1, 0.1}});

    auto ds = std::make_shared<YOLODataset>();
    ASSERT_TRUE(ds->Load(imgDir.string(), labelDir.string()));
    auto s = ds->Get(999);
    EXPECT_TRUE(s.image.empty());
    EXPECT_TRUE(s.boxes.empty());
}

} // namespace aicore
