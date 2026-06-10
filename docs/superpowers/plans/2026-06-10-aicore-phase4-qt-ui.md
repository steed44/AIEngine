# Phase 4 — Qt 上位机实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 Qt5 的精简上位机 AICoreUI.exe，调用 aicore.dll C API 进行单张图片推理并显示检测框。

**Architecture:** 单个 MainWindow（左侧 QLabel 显示图片+检测框，右侧 QTextEdit 显示 JSON）。推理通过 QtConcurrent::run 后台执行，完成后通过信号/槽更新 UI。

**Tech Stack:** C++17, Qt5.12 Widgets+Concurrent, CMake, aicore.dll

---

### Task 1: 创建文件结构和 CMake 集成

**Files:**
- Modify: `aicore/CMakeLists.txt`
- Create: `aicore/gui/main.cpp`
- Create: `aicore/gui/main_window.h`
- Create: `aicore/gui/main_window.cpp`

- [ ] **Step 1: 在 aicore/ 下创建 gui/ 目录**

Run: `New-Item -ItemType Directory -Path "aicore/gui" -Force`

- [ ] **Step 2: 修改 CMakeLists.txt 添加 Qt5 和 AICoreUI 目标**

在 `find_package(GTest REQUIRED)` 之后添加 Qt5 查找。在测试之后添加 AICoreUI 可执行文件。

```cmake
# ──────────────────────────────────────────────
# Qt5 上位机 (AICoreUI.exe)
# ──────────────────────────────────────────────
set(CMAKE_AUTOMOC ON)
find_package(Qt5 REQUIRED COMPONENTS Widgets Concurrent)

add_executable(AICoreUI
    gui/main.cpp
    gui/main_window.cpp
)
target_include_directories(AICoreUI PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(AICoreUI PRIVATE Qt5::Widgets Qt5::Concurrent aicore)
```

- [ ] **Step 3: 创建 main.cpp 入口**

```cpp
#include <QApplication>
#include "main_window.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("AICore UI");
    MainWindow w;
    w.resize(1200, 800);
    w.show();
    return app.exec();
}
```

- [ ] **Step 4: 创建 main_window.h 声明**

```cpp
#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QTextEdit>
#include <QScrollArea>
#include <QPixmap>
#include <QFutureWatcher>
#include "api/aicore_api.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void onOpenImage();
    void onInferenceFinished();

private:
    void initPipeline();
    void drawDetections(QImage& image, const QString& json);
    void showError(const QString& msg);

    QLabel* imageLabel_;
    QScrollArea* scrollArea_;
    QTextEdit* resultText_;
    QAction* openAction_ = nullptr;
    AICorePipeline pipeline_ = nullptr;
    QFutureWatcher<void>* watcher_ = nullptr;
    QImage originalImage_;
    QString lastJson_;
};
```

- [ ] **Step 5: 创建 main_window.cpp 实现（骨架）**

```cpp
#include "main_window.h"
#include <QMenuBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QHBoxLayout>
#include <QSplitter>
#include <QPainter>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QtConcurrent>
#include <fstream>
#include <sstream>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
    setWindowTitle("AICore - 工业检测推理演示");

    auto* menu = menuBar()->addMenu("文件");
    openAction_ = menu->addAction("打开图片 (Ctrl+O)", this, &MainWindow::onOpenImage, QKeySequence::Open);
    menu->addSeparator();
    menu->addAction("退出", this, &QWidget::close);

    imageLabel_ = new QLabel("请打开一张图片");
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setMinimumSize(640, 480);
    scrollArea_ = new QScrollArea;
    scrollArea_->setWidget(imageLabel_);
    scrollArea_->setWidgetResizable(true);

    resultText_ = new QTextEdit;
    resultText_->setReadOnly(true);
    resultText_->setMinimumWidth(300);
    resultText_->setPlaceholderText("检测结果...");

    auto* splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(scrollArea_);
    splitter->addWidget(resultText_);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 1);

    setCentralWidget(splitter);

    watcher_ = new QFutureWatcher<void>(this);
    connect(watcher_, &QFutureWatcher<void>::finished, this, &MainWindow::onInferenceFinished);

    initPipeline();
}

MainWindow::~MainWindow() {
    if (pipeline_) {
        aicore_pipeline_destroy(pipeline_);
        pipeline_ = nullptr;
    }
}

void MainWindow::initPipeline() {
    std::string configStr;
    std::ifstream file("pipeline_config.json");
    if (file) {
        std::stringstream ss;
        ss << file.rdbuf();
        configStr = ss.str();
    } else {
        configStr = R"({
            "pipeline": {
                "name": "default",
                "max_concurrency": 1,
                "enable_profiling": true,
                "nodes": [{"id": "input", "type": "input", "params": {}}],
                "edges": []
            }
        })";
    }
    const char* err = nullptr;
    pipeline_ = aicore_pipeline_create(configStr.c_str(), &err);
    if (!pipeline_) {
        showError(QString("流水线创建失败: %1").arg(err ? err : "未知错误"));
        openAction_->setEnabled(false);
    }
}

void MainWindow::showError(const QString& msg) {
    QMessageBox::warning(this, "错误", msg);
}

void MainWindow::onOpenImage() {
    auto path = QFileDialog::getOpenFileName(this, "打开图片", "",
        "图片 (*.png *.jpg *.jpeg *.bmp *.tiff)");
    if (path.isEmpty()) return;

    originalImage_ = QImage(path);
    if (originalImage_.isNull()) {
        showError("无法加载图片: " + path);
        return;
    }

    if (!pipeline_) {
        showError("流水线未就绪，无法推理");
        return;
    }

    // 显示原图
    imageLabel_->setPixmap(QPixmap::fromImage(originalImage_));

    QImage rgb = originalImage_.convertToFormat(QImage::Format_RGB888);
    int w = rgb.width(), h = rgb.height();
    int channels = 3;

    // 拷贝到堆数据，确保后台线程安全
    auto* data = new uchar[rgb.sizeInBytes()];
    memcpy(data, rgb.bits(), rgb.sizeInBytes());

    resultText_->setText("推理中...");
    lastJson_.clear();

    auto future = QtConcurrent::run([this, data, w, h, channels]() {
        AICoreResult result = nullptr;
        const char* err = nullptr;
        int ret = aicore_pipeline_execute(pipeline_, data, w, h, channels, &result, &err);
        if (ret == 0 && result) {
            const char* json = aicore_result_to_json(result);
            lastJson_ = QString::fromUtf8(json);
            aicore_result_free(result);
        } else {
            lastJson_ = QString("{\"error\":\"%1\"}").arg(err ? err : "推理失败");
        }
        delete[] data;
    });
    watcher_->setFuture(future);
}

void MainWindow::onInferenceFinished() {
    if (lastJson_.isEmpty()) return;

    resultText_->setText(lastJson_);
    drawDetections(originalImage_, lastJson_);
}

void MainWindow::drawDetections(QImage& image, const QString& json) {
    QJsonDocument doc = QJsonDocument::fromJson(json.toUtf8());
    if (doc.isNull() || !doc.isObject()) return;

    QJsonObject root = doc.object();
    QJsonArray detections = root["detections"].toArray();
    if (detections.isEmpty()) return;

    QPainter painter(&image);
    painter.setPen(QPen(Qt::red, 3));
    QFont font = painter.font();
    font.setPixelSize(18);
    painter.setFont(font);

    for (const auto& det : detections) {
        QJsonObject d = det.toObject();
        QJsonObject bbox = d["bbox"].toObject();
        int x = bbox["x"].toInt();
        int y = bbox["y"].toInt();
        int w = bbox["w"].toInt();
        int h = bbox["h"].toInt();

        painter.drawRect(x, y, w, h);

        QString label = QString("%1 %2%")
            .arg(d["label"].toString())
            .arg(static_cast<int>(d["confidence"].toDouble() * 100));
        painter.drawText(QPoint(x, y - 5), label);
    }
    painter.end();

    imageLabel_->setPixmap(QPixmap::fromImage(image));
}
```

- [ ] **Step 6: 编译验证**

Run:
```bash
cd aicore
cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH="C:/Qt/Qt5.12.11/5.12.11/msvc2017_64"
cmake --build build --config Release
```

Expected: AICoreUI.exe built successfully in `build/Release/`
