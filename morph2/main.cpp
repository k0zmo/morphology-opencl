#include <QApplication>
#include <QStyleFactory>

#include <QSettings>
#include <cstdlib>
#include "controller.h"

#ifdef _WIN32
#define setenv(name, value, replace) _putenv_s(name, value)
#endif

int main(int argc, char *argv[])
{
	QApplication::setAttribute(Qt::AA_X11InitThreads);
	QApplication a(argc, argv);

	QStringList styles = QStyleFactory::keys();
	if(styles.contains("QtCurve"))
	{
		QSettings s("./settings.cfg", QSettings::IniFormat);
		QString env = s.value("gui/qtcurvestyle").toString();
		QByteArray data = env.toAscii();

		setenv("QTCURVE_CONFIG_FILE", data.constData(), 1);

		QStyle* style = QStyleFactory::create("QtCurve");
		if(style)
			QApplication::setStyle(style);
	}

	Controller c;
	c.start();

	return a.exec();
}
