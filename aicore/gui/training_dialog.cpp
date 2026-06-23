#include "training_dialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QtConcurrent>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include "patchcore/roi_trainer.h"
#include "trainer/trainer_api.h"

TrainingDialog::TrainingDialog(QWidget* parent)
    : QDialog(parent) {
    setWindowTitle("训练配置");
    resize(700, 650);
    buildUi();
}

TrainingDialog::~TrainingDialog() {
    if (watcher_) {
        watcher_->waitForFinished();
    }
}

void TrainingDialog::buildUi() {
    auto* mainLayout = new QVBoxLayout(this);

    // Tab widget
    tabWidget_ = new QTabWidget(this);
    auto* pcTab = new QWidget;
    auto* yoloTab = new QWidget;
    buildPatchCoreTab(pcTab);
    buildYoloTab(yoloTab);
    tabWidget_->addTab(pcTab, "PatchCore 训练");
    tabWidget_->addTab(yoloTab, "YOLO 训练");
    mainLayout->addWidget(tabWidget_);

    // Log output
    logOutput_ = new QTextEdit(this);
    logOutput_->setReadOnly(true);
    logOutput_->setPlaceholderText("训练日志...");
    logOutput_->setMaximumHeight(200);
    mainLayout->addWidget(logOutput_);

    // Progress bar
    progressBar_ = new QProgressBar(this);
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    mainLayout->addWidget(progressBar_);

    // Buttons
    auto* btnLayout = new QHBoxLayout;
    startBtn_ = new QPushButton("启动训练", this);
    stopBtn_ = new QPushButton("停止", this);
    stopBtn_->setEnabled(false);
    btnLayout->addStretch();
    btnLayout->addWidget(startBtn_);
    btnLayout->addWidget(stopBtn_);
    mainLayout->addLayout(btnLayout);

    // Connections
    connect(startBtn_, &QPushButton::clicked, this, &TrainingDialog::onStartTraining);
    connect(stopBtn_, &QPushButton::clicked, this, &TrainingDialog::onStopTraining);

    // Progress timer (polls progress file every 500ms)
    progressTimer_ = new QTimer(this);
    connect(progressTimer_, &QTimer::timeout, this, &TrainingDialog::onProgressTick);

    // Future watcher
    watcher_ = new QFutureWatcher<void>(this);
    connect(watcher_, &QFutureWatcher<void>::finished, this, &TrainingDialog::onTrainingFinished);
}

void TrainingDialog::buildPatchCoreTab(QWidget* tab) {
    auto* layout = new QVBoxLayout(tab);

    // Data folder
    auto* dataRow = new QHBoxLayout;
    pcDataDir_ = new QLineEdit;
    pcDataDir_->setPlaceholderText("选择训练图片文件夹...");
    auto* dataBtn = new QPushButton("浏览...");
    connect(dataBtn, &QPushButton::clicked, this, [this]() {
        browseDir("选择训练图片文件夹", pcDataDir_);
    });
    dataRow->addWidget(new QLabel("数据文件夹:"));
    dataRow->addWidget(pcDataDir_);
    dataRow->addWidget(dataBtn);
    layout->addLayout(dataRow);

    // Config file (multi-ROI JSON)
    auto* cfgRow = new QHBoxLayout;
    pcConfigFile_ = new QLineEdit;
    pcConfigFile_->setPlaceholderText("可选: ROI 配置文件 (.json)");
    auto* cfgBtn = new QPushButton("浏览...");
    connect(cfgBtn, &QPushButton::clicked, this, [this]() {
        browseFile("选择 ROI 配置文件", "JSON (*.json)", pcConfigFile_);
    });
    cfgRow->addWidget(new QLabel("ROI 配置:"));
    cfgRow->addWidget(pcConfigFile_);
    cfgRow->addWidget(cfgBtn);
    layout->addLayout(cfgRow);

    // Model output dir
    auto* outRow = new QHBoxLayout;
    pcModelDir_ = new QLineEdit("models");
    auto* outBtn = new QPushButton("浏览...");
    connect(outBtn, &QPushButton::clicked, this, [this]() {
        browseDir("选择模型输出目录", pcModelDir_);
    });
    outRow->addWidget(new QLabel("输出目录:"));
    outRow->addWidget(pcModelDir_);
    outRow->addWidget(outBtn);
    layout->addLayout(outRow);

    // Backend type
    auto* backendRow = new QHBoxLayout;
    pcBackendType_ = new QComboBox;
    pcBackendType_->addItems({"libtorch", "opencv_dnn", "model_backend"});
    backendRow->addWidget(new QLabel("Backbone 类型:"));
    backendRow->addWidget(pcBackendType_);
    backendRow->addStretch();
    layout->addLayout(backendRow);

    // Input size, layers, coreset
    auto* paramRow = new QHBoxLayout;
    pcInputSize_ = new QSpinBox;
    pcInputSize_->setRange(64, 1024);
    pcInputSize_->setValue(224);
    pcLayers_ = new QLineEdit("layer2,layer3");
    pcCoreset_ = new QDoubleSpinBox;
    pcCoreset_->setRange(0.01, 1.0);
    pcCoreset_->setSingleStep(0.05);
    pcCoreset_->setValue(0.1);
    pcForceStream_ = new QCheckBox("强制流式模式");
    paramRow->addWidget(new QLabel("输入尺寸:"));
    paramRow->addWidget(pcInputSize_);
    paramRow->addWidget(new QLabel("特征层:"));
    paramRow->addWidget(pcLayers_);
    paramRow->addWidget(new QLabel("Coreset:"));
    paramRow->addWidget(pcCoreset_);
    paramRow->addWidget(pcForceStream_);
    layout->addLayout(paramRow);

    layout->addStretch();
}

void TrainingDialog::buildYoloTab(QWidget* tab) {
    auto* layout = new QVBoxLayout(tab);

    // Data yaml
    auto* dataRow = new QHBoxLayout;
    yDataConfig_ = new QLineEdit;
    yDataConfig_->setPlaceholderText("YOLO 数据配置文件 (.yaml)...");
    auto* dataBtn = new QPushButton("浏览...");
    connect(dataBtn, &QPushButton::clicked, this, [this]() {
        browseFile("选择数据配置", "YAML (*.yaml *.yml)", yDataConfig_);
    });
    dataRow->addWidget(new QLabel("数据配置:"));
    dataRow->addWidget(yDataConfig_);
    dataRow->addWidget(dataBtn);
    layout->addLayout(dataRow);

    // Model
    auto* modelRow = new QHBoxLayout;
    yModelPath_ = new QLineEdit("yolov8n.pt");
    auto* modelBtn = new QPushButton("浏览...");
    connect(modelBtn, &QPushButton::clicked, this, [this]() {
        browseFile("选择预训练模型", "PyTorch (*.pt)", yModelPath_);
    });
    modelRow->addWidget(new QLabel("预训练模型:"));
    modelRow->addWidget(yModelPath_);
    modelRow->addWidget(modelBtn);
    layout->addLayout(modelRow);

    // Hyperparams
    auto* paramGroup = new QGroupBox("超参数");
    auto* paramForm = new QFormLayout(paramGroup);

    yEpochs_ = new QSpinBox;
    yEpochs_->setRange(1, 10000);
    yEpochs_->setValue(100);
    paramForm->addRow("Epochs:", yEpochs_);

    yImgsz_ = new QSpinBox;
    yImgsz_->setRange(32, 4096);
    yImgsz_->setSingleStep(32);
    yImgsz_->setValue(640);
    paramForm->addRow("Imgsz:", yImgsz_);

    yBatch_ = new QSpinBox;
    yBatch_->setRange(1, 1024);
    yBatch_->setValue(16);
    paramForm->addRow("Batch:", yBatch_);

    yDevice_ = new QLineEdit("0");
    paramForm->addRow("Device:", yDevice_);

    yOptimizer_ = new QComboBox;
    yOptimizer_->addItems({"Adam", "SGD", "AdamW"});
    paramForm->addRow("Optimizer:", yOptimizer_);

    yLr0_ = new QDoubleSpinBox;
    yLr0_->setRange(0.00001, 1.0);
    yLr0_->setSingleStep(0.0001);
    yLr0_->setDecimals(5);
    yLr0_->setValue(0.001);
    paramForm->addRow("Learning Rate:", yLr0_);

    yProject_ = new QLineEdit("runs/train");
    paramForm->addRow("Project:", yProject_);

    yName_ = new QLineEdit("exp");
    paramForm->addRow("Name:", yName_);

    layout->addWidget(paramGroup);
    layout->addStretch();
}

QString TrainingDialog::browseDir(const QString& title, QLineEdit* target) {
    auto dir = QFileDialog::getExistingDirectory(this, title, target->text());
    if (!dir.isEmpty()) {
        target->setText(dir);
    }
    return dir;
}

QString TrainingDialog::browseFile(const QString& title, const QString& filter, QLineEdit* target) {
    auto path = QFileDialog::getOpenFileName(this, title, target->text(), filter);
    if (!path.isEmpty()) {
        target->setText(path);
    }
    return path;
}

void TrainingDialog::appendLog(const QString& msg) {
    auto ts = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    logOutput_->append(QString("[%1] %2").arg(ts, msg));
}

void TrainingDialog::setTrainingEnabled(bool enabled) {
    startBtn_->setEnabled(!enabled);
    stopBtn_->setEnabled(enabled);
    tabWidget_->setEnabled(!enabled);
    if (!enabled) {
        progressBar_->setValue(0);
    }
}

void TrainingDialog::onStartTraining() {
    int idx = tabWidget_->currentIndex();
    if (idx == 0) {
        onStartPatchCore();
    } else {
        onStartYolo();
    }
}

void TrainingDialog::onStartPatchCore() {
    if (pcDataDir_->text().isEmpty()) {
        QMessageBox::warning(this, "参数错误", "请选择数据文件夹");
        return;
    }

    setTrainingEnabled(true);
    stopRequested_ = false;
    progressBar_->setRange(0, 0); // indeterminate
    appendLog("PatchCore 训练开始...");

    trainException_ = nullptr;
    auto configPath = pcConfigFile_->text().toStdString();
    auto dataFolder = pcDataDir_->text().toStdString();
    auto modelDir = pcModelDir_->text().toStdString();
    auto forceStream = pcForceStream_->isChecked();
    auto backendType = pcBackendType_->currentText().toStdString();
    auto inputSize = pcInputSize_->value();
    auto layers = pcLayers_->text().toStdString();
    auto coreset = pcCoreset_->value();

    auto future = QtConcurrent::run([=]() {
        try {
            if (!configPath.empty() && QFile::exists(QString::fromStdString(configPath))) {
                aicore::RoiTrainer trainer;
                trainer.SetForceStream(forceStream);
                trainer.SetForceNoStream(false);
                auto s = trainer.TrainAll(configPath, dataFolder);
                if (!s) throw std::runtime_error(s.message);
            } else {
                nlohmann::json j;
                j["rois"] = nlohmann::json::array();
                nlohmann::json roi;
                roi["name"] = "roi";
                roi["data_dir"] = dataFolder;
                roi["model_path"] = "";
                roi["backend_type"] = backendType;
                roi["input_size"] = inputSize;
                roi["backbone_layers"] = layers;
                roi["coreset_fraction"] = coreset;
                roi["output_path"] = modelDir + "/memory_bank.bin";
                j["rois"].push_back(roi);

                auto tmpCfg = std::filesystem::temp_directory_path() / "aicore_roi_train.json";
                std::ofstream f(tmpCfg.string());
                f << j.dump(4);
                f.close();

                aicore::RoiTrainer trainer;
                trainer.SetForceStream(forceStream);
                trainer.SetForceNoStream(false);
                auto s = trainer.TrainAll(tmpCfg.string(), dataFolder);
                if (!s) throw std::runtime_error(s.message);
            }
        } catch (...) {
            trainException_ = std::current_exception();
        }
    });

    watcher_->setFuture(future);
}

void TrainingDialog::onStartYolo() {
    if (yDataConfig_->text().isEmpty()) {
        QMessageBox::warning(this, "参数错误", "请选择数据配置文件");
        return;
    }

    trainException_ = nullptr;
    setTrainingEnabled(true);
    stopRequested_ = false;
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    appendLog("YOLO 训练开始...");

    nlohmann::json cfg;
    cfg["data"] = yDataConfig_->text().toStdString();
    cfg["model"] = yModelPath_->text().toStdString();
    cfg["epochs"] = yEpochs_->value();
    cfg["imgsz"] = yImgsz_->value();
    cfg["batch"] = yBatch_->value();
    cfg["device"] = yDevice_->text().toStdString();
    cfg["optimizer"] = yOptimizer_->currentText().toStdString();
    cfg["lr0"] = yLr0_->value();
    cfg["project"] = yProject_->text().toStdString();
    cfg["name"] = yName_->text().toStdString();
    cfg["pretrained"] = true;

    progressFile_ = (std::filesystem::temp_directory_path() / "aicore_progress.jsonl").string();
    std::filesystem::remove(progressFile_);

    auto configJson = cfg.dump();
    auto future = QtConcurrent::run([this, configJson]() {
        try {
            const char* err = nullptr;
            int ret = aicore_train_run(configJson.c_str(), &err);
            if (ret != 0) {
                throw std::runtime_error(err ? err : "YOLO training failed");
            }
        } catch (...) {
            trainException_ = std::current_exception();
        }
    });

    progressTimer_->start(500);
    watcher_->setFuture(future);
}

void TrainingDialog::onStopTraining() {
    stopRequested_ = true;
    appendLog("正在停止训练...");
}

void TrainingDialog::onProgressTick() {
    if (progressFile_.empty()) return;

    QFile file(QString::fromStdString(progressFile_));
    if (!file.open(QIODevice::ReadOnly)) return;

    QTextStream in(&file);
    QString lastLine;
    while (!in.atEnd()) {
        lastLine = in.readLine();
    }
    file.close();

    if (lastLine.isEmpty()) return;

    try {
        auto j = nlohmann::json::parse(lastLine.toStdString());
        if (j.contains("epoch") && j.contains("total_epochs")) {
            int epoch = j["epoch"];
            int total = j["total_epochs"];
            progressBar_->setValue(static_cast<int>(100.0 * epoch / total));
        }
        if (j.contains("loss")) {
            appendLog(QString("Epoch %1/%2, loss: %3")
                .arg(j.value("epoch", 0))
                .arg(j.value("total_epochs", 0))
                .arg(j.value("loss", 0.0)));
        }
    } catch (...) {
        // ignore non-JSON lines
    }
}

void TrainingDialog::onTrainingFinished() {
    progressTimer_->stop();
    setTrainingEnabled(false);

    // Clean up progress file
    if (!progressFile_.empty()) {
        std::filesystem::remove(progressFile_);
        progressFile_.clear();
    }

    if (trainException_) {
        try {
            std::rethrow_exception(trainException_);
        } catch (const std::exception& e) {
            appendLog(QString("训练失败: %1").arg(e.what()));
            QMessageBox::warning(this, "训练失败", e.what());
        }
        trainException_ = nullptr;
    } else {
        appendLog("训练完成！");
        QMessageBox::information(this, "训练完成", "模型训练已成功完成。");
    }
}
