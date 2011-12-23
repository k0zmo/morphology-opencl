#include "mainwindow.h"
#include <QtGui/QApplication>
#include <QSettings>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	QSettings settings("./settings.cfg", QSettings::IniFormat);
	QString filename = settings.value("gui/defaultimage", "").toString();

	if(filename.isEmpty())
	{
		filename = QFileDialog::getOpenFileName(
			nullptr, QString(), ".",
			QString::fromLatin1("Image files (*.png *.jpg *.bmp)"));

		if(filename.isEmpty()) 
		{
			a.quit();
			return 0;
		}
	}

	MainWindow w(filename);
	w.show();
	return a.exec();
}
