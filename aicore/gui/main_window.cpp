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

    imageLabel_->setPixmap(QPixmap::fromImage(originalImage_));

    QImage rgb = originalImage_.convertToFormat(QImage::Format_RGB888);
    int w = rgb.width(), h = rgb.height();
    int channels = 3;

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
    drawAnomalyOverlay(originalImage_, lastJson_);
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
    painter.fillRect(image.rect(),
        QColor(255, 0, 0, static_cast<int>(std::min(score * 200, 100.0))));

    QFont font = painter.font();
    font.setPixelSize(20);
    font.setBold(true);
    painter.setFont(font);

    QString text = QString("Anomaly Score: %1").arg(score, 0, 'f', 4);
    painter.setPen(isAnomaly ? Qt::red : Qt::green);
    painter.drawText(QPoint(10, 30), text);

    if (isAnomaly) {
        painter.setPen(QPen(Qt::red, 3));
        painter.drawRect(5, 5, image.width() - 10, image.height() - 10);
    }
    painter.end();
}
