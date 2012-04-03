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

typedef QMap<oclPlatformDesc, QList<oclDeviceDesc> > PlatformDevicesMap;

inline bool operator<(const oclPlatformDesc& p1, const oclPlatformDesc& p2)
{
	return p1.id < p2.id;
}

class oclThread : public QThread
{
	Q_OBJECT
public:
	oclThread(BlockingQueue<ProcessingItem>& queue,
		const Configuration& conf);
	virtual ~oclThread();

	PlatformDevicesMap queryPlatforms();
	void choose(int platformId, int deviceId);
	void stop();
	virtual void run();

private:
	BlockingQueue<ProcessingItem>& queue;
	QMutex stopMutex;

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
