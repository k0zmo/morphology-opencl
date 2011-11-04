#include "mainwindow.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

#if 0
	QString filename = QFileDialog::getOpenFileName(
		nullptr, QString(), ".",
		QString::fromLatin1("Image files (*.png *.jpg *.bmp)"));

	if(filename.isEmpty()) {
		a.quit();
		return 0;
	}
#else
	QSettings settings("./settings.cfg", QSettings::IniFormat);
	QString filename = settings.value("gui/defaultimage", "lena.jpg").toString();
#endif


	MainWindow w(filename);
	w.show();
	return a.exec();
}
