// ============================================================
// 文件: gui/main.cpp
// 用途: AICore GUI 应用程序入口点
// ============================================================

#include <QApplication>
#include "main_window.h"

// 程序入口：初始化 Qt 应用程序并显示主窗口
int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("AICore UI");
    MainWindow w;
    w.resize(1200, 800);   // 设置默认窗口尺寸 1200x800
    w.show();
    return app.exec();      // 进入 Qt 事件循环
}
