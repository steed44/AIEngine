// 训练对话框 — PatchCore / YOLOv8 训练 UI 入口
// 提供训练参数配置、进度监控和日志输出的桌面界面
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
#include <mutex>
#include <string>

// 训练对话框
// 通过选项卡切换 PatchCore 训练 和 YOLOv8 训练
// 支持训练启停控制、实时日志输出、进度条显示
class TrainingDialog : public QDialog {
    Q_OBJECT
public:
    explicit TrainingDialog(QWidget* parent = nullptr);
    ~TrainingDialog() override;

private slots:
    void onStartTraining();       // 槽：根据当前选项卡启动对应训练
    void onStopTraining();        // 槽：请求停止当前训练
    void onTrainingFinished();    // 槽：训练完成后恢复 UI 状态
    void onProgressTick();        // 槽：定时读取进度文件并更新 UI

private:
    QTabWidget* tabWidget_ = nullptr;       // PatchCore / YOLO 选项卡
    QTextEdit* logOutput_ = nullptr;        // 日志输出控件
    QProgressBar* progressBar_ = nullptr;   // 训练进度条
    QPushButton* startBtn_ = nullptr;       // 开始训练按钮
    QPushButton* stopBtn_ = nullptr;        // 停止训练按钮
    QTimer* progressTimer_ = nullptr;       // 进度轮询定时器
    QFutureWatcher<void>* watcher_ = nullptr; // 异步训练任务监视器

    std::atomic<bool> stopRequested_{false};  // 停止请求标志（跨线程安全）
    std::string progressFile_;                // 进度文件路径（Python 训练用）
    std::exception_ptr trainException_;       // 训练线程异常（主线程检查）
    std::mutex progressMutex_;                // 进度数据互斥锁

    // ---- PatchCore 训练参数 UI 控件 ----
    QLineEdit* pcDataDir_ = nullptr;          // 正常样本目录
    QLineEdit* pcConfigFile_ = nullptr;       // ROI 配置文件路径
    QLineEdit* pcModelDir_ = nullptr;         // 模型输出目录
    QComboBox* pcBackendType_ = nullptr;      // backbone 类型
    QSpinBox* pcInputSize_ = nullptr;         // 输入尺寸
    QLineEdit* pcLayers_ = nullptr;           // backbone 层名
    QDoubleSpinBox* pcCoreset_ = nullptr;     // Coreset 比例
    QCheckBox* pcForceStream_ = nullptr;      // 强制流式训练

    // ---- YOLOv8 训练参数 UI 控件（原生 C++ YOLOTrainer） ----
    QLineEdit* yTrainImgDir_ = nullptr;       // 训练图片目录
    QLineEdit* yTrainLabelDir_ = nullptr;     // 训练标签目录
    QSpinBox* yNumClasses_ = nullptr;         // 类别数
    QLineEdit* yModelPath_ = nullptr;         // 模型权重路径
    QSpinBox* yEpochs_ = nullptr;             // 训练轮数
    QSpinBox* yImgsz_ = nullptr;              // 输入尺寸
    QSpinBox* yBatch_ = nullptr;              // 批次大小
    QLineEdit* yDevice_ = nullptr;            // 设备（如 "0" 或 "cpu"）
    QComboBox* yOptimizer_ = nullptr;         // 优化器选择
    QDoubleSpinBox* yLr0_ = nullptr;          // 初始学习率
    QLineEdit* yProject_ = nullptr;           // 项目保存目录
    QLineEdit* yName_ = nullptr;              // 实验名称

    void buildUi();                     // 构建对话框布局
    void buildPatchCoreTab(QWidget* tab); // 构建 PatchCore 参数选项卡
    void buildYoloTab(QWidget* tab);      // 构建 YOLO 参数选项卡
    QString browseDir(const QString& title, QLineEdit* target);   // 目录选择对话框
    QString browseFile(const QString& title, const QString& filter, QLineEdit* target); // 文件选择对话框
    void appendLog(const QString& msg);   // 追加日志到输出框
    void setTrainingEnabled(bool enabled); // 切换训练启停 UI 状态
    void onStartPatchCore();              // 启动 PatchCore 训练
    void onStartYolo();                   // 启动 YOLO 训练
};
