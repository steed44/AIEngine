#pragma once
#include <QDialog>
#include <QTabWidget>
#include <QTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QLineEdit>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QTimer>
#include <QFutureWatcher>
#include <atomic>
#include <string>

class TrainingDialog : public QDialog {
    Q_OBJECT
public:
    explicit TrainingDialog(QWidget* parent = nullptr);
    ~TrainingDialog() override;

private slots:
    void onStartTraining();
    void onStopTraining();
    void onTrainingFinished();
    void onProgressTick();

private:
    QTabWidget* tabWidget_ = nullptr;
    QTextEdit* logOutput_ = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QPushButton* startBtn_ = nullptr;
    QPushButton* stopBtn_ = nullptr;
    QTimer* progressTimer_ = nullptr;
    QFutureWatcher<void>* watcher_ = nullptr;

    std::atomic<bool> stopRequested_{false};
    std::string progressFile_;
    std::exception_ptr trainException_;

    // PatchCore tab
    QLineEdit* pcDataDir_ = nullptr;
    QLineEdit* pcConfigFile_ = nullptr;
    QLineEdit* pcModelDir_ = nullptr;
    QComboBox* pcBackendType_ = nullptr;
    QSpinBox* pcInputSize_ = nullptr;
    QLineEdit* pcLayers_ = nullptr;
    QDoubleSpinBox* pcCoreset_ = nullptr;
    QCheckBox* pcForceStream_ = nullptr;

    // YOLO tab
    QLineEdit* yDataConfig_ = nullptr;
    QLineEdit* yModelPath_ = nullptr;
    QSpinBox* yEpochs_ = nullptr;
    QSpinBox* yImgsz_ = nullptr;
    QSpinBox* yBatch_ = nullptr;
    QLineEdit* yDevice_ = nullptr;
    QComboBox* yOptimizer_ = nullptr;
    QDoubleSpinBox* yLr0_ = nullptr;
    QLineEdit* yProject_ = nullptr;
    QLineEdit* yName_ = nullptr;

    void buildUi();
    void buildPatchCoreTab(QWidget* tab);
    void buildYoloTab(QWidget* tab);
    QString browseDir(const QString& title, QLineEdit* target);
    QString browseFile(const QString& title, const QString& filter, QLineEdit* target);
    void appendLog(const QString& msg);
    void setTrainingEnabled(bool enabled);
    void onStartPatchCore();
    void onStartYolo();
};
