#pragma once

#include <QThread>
#include <QMap>
#include <QList>

#include "blockingqueue.h"
#include "configuration.h"

#include <qclplatform.h>
#include <qcldevice.h>
class QCLContext;
class QCLContextGL;
class GLDummyWidget;

// Lista pobranych danych o platformach i ich urzadzeniach
typedef QList<QPair<QCLPlatform, QList<QCLDevice>>> PlatformDevicesList;

enum EOpenCLBackend
{
	OB_Images,
	OB_Buffers
};

class oclThread : public QThread
{
	Q_OBJECT
public:
	oclThread(BlockingQueue<ProcessingItem>& queue,
		const Configuration& conf, GLDummyWidget* shareWidget = nullptr);
	virtual ~oclThread();

	// Mozna wolac tylko przed start()
	void setSharedWidget(GLDummyWidget* shareWidget);
	void setOpenCLBackend(EOpenCLBackend backend);

	PlatformDevicesList queryPlatforms();
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
	EOpenCLBackend backend;

	Configuration conf;

private:
	void initContext(QCLContext* ctx);
	void initContextWithGL(QCLContextGL* ctx);

	void startWithImages();
	void startWithBuffers();

signals:
	void processingDone(const ProcessedItem& item);
	void openCLInitialized(bool success);
};
