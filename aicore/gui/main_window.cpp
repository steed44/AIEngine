// ============================================================
// 文件: gui/main_window.cpp
// 用途: AICore 主窗口实现 — Qt5 桌面推理前端
// ============================================================

#include "main_window.h"
#include "training_dialog.h"
#include "api/scheduler_api.h"
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
#include <QActionGroup>
#include <fstream>
#include <sstream>
#include <vector>

// 构造函数：设置菜单栏、图片显示区、结果文本区和信号槽连接
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
    setWindowTitle("AICore - 工业检测推理演示");

    // --- 菜单栏：文件菜单 ---
    auto* menu = menuBar()->addMenu("文件");
    openAction_ = menu->addAction("打开图片 (Ctrl+O)", this, &MainWindow::onOpenImage, QKeySequence::Open);
    menu->addSeparator();
    menu->addAction("退出", this, &QWidget::close);

    // --- 菜单栏：训练菜单 ---
    auto* trainMenu = menuBar()->addMenu("训练");
    trainMenu->addAction("训练配置...", this, &MainWindow::onTrainingDialog, QKeySequence("Ctrl+T"));

    // --- 菜单栏：GPU 调度 ---
    auto* schedMenu = menuBar()->addMenu("GPU 调度");
    auto* schedGroup = new QActionGroup(this);
    inferPriorityAction_ = schedMenu->addAction("推理优先");
    inferPriorityAction_->setCheckable(true);
    inferPriorityAction_->setChecked(true);
    schedGroup->addAction(inferPriorityAction_);
    trainPriorityAction_ = schedMenu->addAction("训练优先");
    trainPriorityAction_->setCheckable(true);
    schedGroup->addAction(trainPriorityAction_);
    balancedPriorityAction_ = schedMenu->addAction("均衡");
    balancedPriorityAction_->setCheckable(true);
    schedGroup->addAction(balancedPriorityAction_);
    connect(inferPriorityAction_, &QAction::triggered, this, [this]{ onSchedulerPriority(0); });
    connect(trainPriorityAction_, &QAction::triggered, this, [this]{ onSchedulerPriority(1); });
    connect(balancedPriorityAction_, &QAction::triggered, this, [this]{ onSchedulerPriority(2); });

    // --- 图片显示区域（带滚动） ---
    imageLabel_ = new QLabel("请打开一张图片");
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setMinimumSize(640, 480);
    scrollArea_ = new QScrollArea;
    scrollArea_->setWidget(imageLabel_);
    scrollArea_->setWidgetResizable(true);

    // --- 结果文本显示区域 ---
    resultText_ = new QTextEdit;
    resultText_->setReadOnly(true);
    resultText_->setMinimumWidth(300);
    resultText_->setPlaceholderText("检测结果...");

    // --- 使用 QSplitter 分割图片区和结果区 ---
    auto* splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(scrollArea_);
    splitter->addWidget(resultText_);
    splitter->setStretchFactor(0, 3);  // 图片区比例 3
    splitter->setStretchFactor(1, 1);  // 结果区比例 1
    setCentralWidget(splitter);

    // --- 异步推理完成信号连接 ---
    watcher_ = new QFutureWatcher<void>(this);
    connect(watcher_, &QFutureWatcher<void>::finished, this, &MainWindow::onInferenceFinished);

    // 初始化推理流水线
    initPipeline();
}

// 析构函数：释放推理流水线资源
MainWindow::~MainWindow() {
    if (pipeline_) {
        aicore_pipeline_destroy(pipeline_);
        pipeline_ = nullptr;
    }
}

// 初始化推理流水线：从 JSON 配置文件加载，若无文件则使用默认配置
void MainWindow::initPipeline() {
    std::string configStr;
    std::ifstream file("pipeline_config.json");
    if (file) {
        std::stringstream ss;
        ss << file.rdbuf();
        configStr = ss.str();
    } else {
        // 默认流水线配置（仅含 input 节点）
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

// 显示错误警告对话框
void MainWindow::showError(const QString& msg) {
    QMessageBox::warning(this, "错误", msg);
}

// 设置 GPU 调度优先级模式
void MainWindow::onSchedulerPriority(int mode) {
    const char* modeStr = "balanced";
    if (mode == 0) modeStr = "inference";
    else if (mode == 1) modeStr = "training";
    aicore_scheduler_set_priority(modeStr);
}

// 打开训练配置对话框
void MainWindow::onTrainingDialog() {
    TrainingDialog dlg(this);
    dlg.exec();
}

// 打开图片文件 → 异步执行推理
void MainWindow::onOpenImage() {
    auto path = QFileDialog::getOpenFileName(this, "打开图片", "",
        "图片 (*.png *.jpg *.jpeg *.bmp *.tiff)");
    if (path.isEmpty()) return;

    // 加载原始图片
    originalImage_ = QImage(path);
    if (originalImage_.isNull()) {
        showError("无法加载图片: " + path);
        return;
    }

    if (!pipeline_) {
        showError("流水线未就绪，无法推理");
        return;
    }

    // 在界面上显示原图
    imageLabel_->setPixmap(QPixmap::fromImage(originalImage_));

    // 转为 RGB888 格式并拷贝像素数据
    QImage rgb = originalImage_.convertToFormat(QImage::Format_RGB888);
    int w = rgb.width(), h = rgb.height();
    int channels = 3;

    std::vector<uchar> data(rgb.sizeInBytes());
    memcpy(data.data(), rgb.bits(), rgb.sizeInBytes());

    // 清理上次结果
    if (lastResult_) { aicore_result_free(lastResult_); lastResult_ = nullptr; }
    resultText_->setText("推理中...");
    lastJson_.clear();

    // 在后台线程中执行推理，避免阻塞 UI
    auto future = QtConcurrent::run([this, data, w, h, channels]() {
        AICoreResult result = nullptr;
        const char* err = nullptr;
        int ret = aicore_pipeline_execute(pipeline_, data.data(), w, h, channels, &result, &err);
        if (ret == 0 && result) {
            const char* json = aicore_result_to_json(result);
            lastJson_ = QString::fromUtf8(json);
            lastResult_ = result;  // 保存 result_ 指针供 onInferenceFinished 使用
        } else {
            lastJson_ = QString("{\"error\":\"%1\"}").arg(err ? err : "推理失败");
        }
    });
    watcher_->setFuture(future);
}

// 槽：推理完成 → 显示 JSON 结果，绘制检测框和异常热力图，释放 result_
void MainWindow::onInferenceFinished() {
    if (lastJson_.isEmpty()) return;

    resultText_->setText(lastJson_);
    drawDetections(originalImage_, lastJson_);
    drawAnomalyOverlay(originalImage_, lastJson_);

    // 释放 result_（drawAnomalyOverlay 中通过 C API 读取了热力图数据）
    if (lastResult_) {
        aicore_result_free(lastResult_);
        lastResult_ = nullptr;
    }
}

// 在图片上绘制检测框和标签（红色矩形 + 类别名/置信度）
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

        // 绘制检测框
        painter.drawRect(x, y, w, h);

        // 绘制标签（类别名 + 置信度百分比）
        QString label = QString("%1 %2%")
            .arg(d["label"].toString())
            .arg(static_cast<int>(d["confidence"].toDouble() * 100));
        painter.drawText(QPoint(x, y - 5), label);
    }
    painter.end();

    imageLabel_->setPixmap(QPixmap::fromImage(image));
}

// 绘制异常检测热力叠加层：从 lastResult_ 读取像素级热力图并渲染 Jet 色彩映射
void MainWindow::drawAnomalyOverlay(QImage& image, const QString& json) {
    QJsonDocument doc = QJsonDocument::fromJson(json.toUtf8());
    if (doc.isNull() || !doc.isObject()) return;

    QJsonObject root = doc.object();
    QJsonArray dets = root["detections"].toArray();
    int detIndex = -1;
    double score = -1;
    for (int i = 0; i < dets.size(); i++) {
        QJsonObject obj = dets[i].toObject();
        if (obj.contains("anomaly_score")) {
            score = obj["anomaly_score"].toDouble();
            detIndex = i;
            break;
        }
    }
    if (score < 0) return;

    bool gotMap = false;
    float* mapData = nullptr;
    int mapW = 0, mapH = 0;

    if (lastResult_) {
        gotMap = (aicore_result_get_anomaly_map(lastResult_, detIndex,
                   &mapData, &mapW, &mapH) == 0 && mapData);
    }

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);

    if (gotMap) {
        // 像素级 Jet 色彩映射叠加
        QImage heatmap(image.size(), QImage::Format_ARGB32);
        heatmap.fill(Qt::transparent);
        float scaleX = (float)mapW / image.width();
        float scaleY = (float)mapH / image.height();

        for (int y = 0; y < image.height(); y++) {
            int my = qMin((int)(y * scaleY), mapH - 1);
            for (int x = 0; x < image.width(); x++) {
                float v = mapData[my * mapW + qMin((int)(x * scaleX), mapW - 1)];
                float t = qBound(0.0f, v / 3.0f, 1.0f);
                int r, g, b;
                if (t < 0.25f)      { r = 0; g = (int)(t * 1024); b = 255; }
                else if (t < 0.5f)  { r = 0; g = 255; b = (int)((0.5f - t) * 1024); }
                else if (t < 0.75f) { r = (int)((t - 0.5f) * 1024); g = 255; b = 0; }
                else                { r = 255; g = (int)((1.0f - t) * 1024); b = 0; }
                int a = qMin((int)(v * 60), 180);
                heatmap.setPixelColor(x, y, QColor(r, g, b, a));
            }
        }
        painter.drawImage(0, 0, heatmap);
        aicore_result_free_anomaly_map(mapData);
    } else {
        // 退化：纯色半透明叠加
        int alpha = qMin((int)(score * 200), 100);
        painter.fillRect(image.rect(), QColor(255, 0, 0, alpha));
    }

    QFont font = painter.font();
    font.setPixelSize(20);
    font.setBold(true);
    painter.setFont(font);
    bool isAnomaly = score > 0.5;
    painter.setPen(isAnomaly ? Qt::red : Qt::green);
    painter.drawText(QPoint(10, 30), QString("Anomaly Score: %1").arg(score, 0, 'f', 4));

    if (isAnomaly) {
        painter.setPen(QPen(Qt::red, 3));
        painter.drawRect(5, 5, image.width() - 10, image.height() - 10);
    }
    painter.end();
}
