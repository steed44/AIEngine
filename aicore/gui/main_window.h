// ============================================================
// 文件: gui/main_window.h
// 用途: AICore 主窗口声明 — 用于图片推理的桌面 GUI
// ============================================================

#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QTextEdit>
#include <QScrollArea>
#include <QAction>
#include <QImage>
#include <QFutureWatcher>
#include <QMenu>
#include "api/aicore_api.h"

// 主窗口类：提供图片加载、推理执行和结果可视化的桌面界面
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void onOpenImage();          // 槽：打开图片并启动推理
    void onInferenceFinished();  // 槽：推理完成后更新界面
    void onTrainingDialog();     // 槽：打开训练对话框

private:
    void initPipeline();                           // 初始化推理流水线
    void drawDetections(QImage& image, const QString& json);     // 在图片上绘制检测框
    void drawAnomalyOverlay(QImage& image, const QString& json); // 绘制异常热力图叠加层
    void showError(const QString& msg);                         // 显示错误对话框

    QLabel* imageLabel_;             // 图片显示标签
    QScrollArea* scrollArea_;        // 可滚动的图片显示区域
    QTextEdit* resultText_;          // 结果文本显示框
    QAction* openAction_ = nullptr;  // "打开图片"菜单动作
    AICorePipeline pipeline_ = nullptr;            // 推理流水线句柄
    QFutureWatcher<void>* watcher_ = nullptr;      // 异步任务监视器
    QImage originalImage_;           // 原始图片副本
    QString lastJson_;               // 上一次推理的 JSON 结果
};
