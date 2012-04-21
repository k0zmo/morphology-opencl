#include <QApplication>
#include <QCleanlooksStyle>

#include "controller.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QApplication::setStyle(new QCleanlooksStyle);
	Controller c;
	c.start();

	return a.exec();
}
