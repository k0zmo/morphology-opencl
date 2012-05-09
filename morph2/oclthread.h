#pragma once

#include <QThread>
#include <QMap>
#include <QList>

#include "blockingqueue.h"
#include "configuration.h"

#include <QCLPlatform>
#include <QCLDevice>
class QCLContext;
class QCLContextGL;
class GLDummyWidget;

// Mapa pobranych danych o platformach i ich urzadzeniach
typedef QMap<QCLPlatform, QList<QCLDevice> > PlatformDevicesMap;
inline bool operator<(const QCLPlatform& p1, const QCLPlatform& p2)
{ return p1.platformId() < p2.platformId(); }

class oclThread : public QThread
{
	Q_OBJECT
public:
	oclThread(BlockingQueue<ProcessingItem>& queue,
		const Configuration& conf, GLDummyWidget* shareWidget = nullptr);
	virtual ~oclThread();

	// Mozna wolac tylko przed start()
	void setSharedWidget(GLDummyWidget* shareWidget);

	PlatformDevicesMap queryPlatforms();
	void choose(int platformId, int deviceId);
	void stop();
	virtual void run();

private:
	BlockingQueue<ProcessingItem>& queue;
	QMutex stopMutex;

	GLDummyWidget* shareWidget;

	int platformId, deviceId;
	bool stopped;
	bool success;

	Configuration conf;

private:
	void initContext(QCLContext* ctx);
	void initContextWithGL(QCLContextGL* ctx);

signals:
	void processingDone(const ProcessedItem& item);
	void openCLInitialized(bool success);
};
