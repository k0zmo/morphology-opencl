#include <QApplication>
#include <QCleanlooksStyle>

#include "controller.h"

int main(int argc, char *argv[])
{
	QApplication::setAttribute(Qt::AA_X11InitThreads);
	QApplication::setStyle(new QCleanlooksStyle);

	QApplication a(argc, argv);
	Controller c;
	c.start();

	return a.exec();
}
