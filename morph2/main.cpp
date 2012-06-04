#include <QApplication>
#include <QStyleFactory>

#include <QSettings>
#include <cstdlib>
#include "controller.h"

#ifdef Q_WS_WIN32
#	define setenv(name, value, replace) _putenv_s(name, value)
#endif

#include "minidumper.h"

int main(int argc, char *argv[])
{
#ifdef Q_WS_WIN32
	MiniDumper dumper("morph2.exe");
#endif

	QApplication::setAttribute(Qt::AA_X11InitThreads);
	QApplication a(argc, argv);

	QFile f(":/UI/gray.qss");
	f.open(QFile::ReadOnly);
	a.setStyleSheet(f.readAll());

	Controller c;
	c.start();

	return a.exec();
}
