#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QTextEdit>
#include <QScrollArea>
#include <QAction>
#include <QImage>
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
    void drawAnomalyOverlay(QImage& image, const QString& json);
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
