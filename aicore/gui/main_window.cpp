// ============================================================
// 文件: gui/main_window.cpp
// 用途: AICore 主窗口实现 — Qt5 桌面推理前端
// ============================================================

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

// 构造函数：设置菜单栏、图片显示区、结果文本区和信号槽连接
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
    setWindowTitle("AICore - 工业检测推理演示");

    // --- 菜单栏：文件菜单 ---
    auto* menu = menuBar()->addMenu("文件");
    openAction_ = menu->addAction("打开图片 (Ctrl+O)", this, &MainWindow::onOpenImage, QKeySequence::Open);
    menu->addSeparator();
    menu->addAction("退出", this, &QWidget::close);

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

// 槽：打开图片文件 → 转换为 RGB 数据 → 异步执行推理
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

    auto* data = new uchar[rgb.sizeInBytes()];
    memcpy(data, rgb.bits(), rgb.sizeInBytes());

    resultText_->setText("推理中...");
    lastJson_.clear();

    // 在后台线程中执行推理，避免阻塞 UI
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

// 槽：推理完成 → 显示 JSON 结果并在图片上绘制检测框和异常叠加层
void MainWindow::onInferenceFinished() {
    if (lastJson_.isEmpty()) return;

    resultText_->setText(lastJson_);           // 显示原始 JSON
    drawDetections(originalImage_, lastJson_); // 绘制检测框
    drawAnomalyOverlay(originalImage_, lastJson_); // 绘制异常热力叠加层
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

// 绘制异常检测热力叠加层：根据 anomaly_score 显示半透明红色叠加
void MainWindow::drawAnomalyOverlay(QImage& image, const QString& json) {
    QJsonDocument doc = QJsonDocument::fromJson(json.toUtf8());
    if (doc.isNull() || !doc.isObject()) return;

    QJsonObject root = doc.object();
    QJsonArray dets = root["detections"].toArray();
    double score = -1;
    for (auto& d : dets) {
        QJsonObject obj = d.toObject();
        if (obj.contains("anomaly_score")) {
            score = obj["anomaly_score"].toDouble();
            break;
        }
    }
    if (score < 0) return;

    bool isAnomaly = score > 0.5;

    QPainter painter(&image);
    // 根据异常分数填充半透明红色（分数越高越深）
    painter.fillRect(image.rect(),
        QColor(255, 0, 0, static_cast<int>(std::min(score * 200, 100.0))));

    QFont font = painter.font();
    font.setPixelSize(20);
    font.setBold(true);
    painter.setFont(font);

    // 左上角显示异常分数
    QString text = QString("Anomaly Score: %1").arg(score, 0, 'f', 4);
    painter.setPen(isAnomaly ? Qt::red : Qt::green);
    painter.drawText(QPoint(10, 30), text);

    if (isAnomaly) {
        // 异常时绘制红色边框
        painter.setPen(QPen(Qt::red, 3));
        painter.drawRect(5, 5, image.width() - 10, image.height() - 10);
    }
    painter.end();
}
