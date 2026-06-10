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
