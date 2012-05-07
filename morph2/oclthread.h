#pragma once

#include <QThread>
#include <QMap>
#include <QList>

#include "blockingqueue.h"
#include "configuration.h"

#include "oclcontext.h"
#include "oclfilter.h"
#include "oclbayerfilter.h"
#include "oclmorphfilter.h"
#include "oclmorphhitmissfilter.h"

class GLDummyWidget;

// Mapa pobranych danych o platformach i ich urzadzeniach
typedef QMap<oclPlatformDesc, QList<oclDeviceDesc> > PlatformDevicesMap;
inline bool operator<(const oclPlatformDesc& p1, const oclPlatformDesc& p2)
{ return p1.id < p2.id; }

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

	oclContext c;
	oclBayerFilter* bayerFilter;
	oclMorphFilter* morphFilter;
	oclMorphHitMissFilter* hitmissFilter;

private:
	void initContext();

signals:
	void processingDone(const ProcessedItem& item);
	void openCLInitialized(bool success);
};
