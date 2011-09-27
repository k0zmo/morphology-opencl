#include "stdafx.h"
#include "morph.h"
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
	//QString filename = "circles.png";
	//QString filename = "LabelingImageBW.png";
	QString filename = "bin4.png";
#endif


	Morph w(filename);
	w.show();
	return a.exec();
}
