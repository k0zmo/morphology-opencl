#pragma once

#include <QThread>

#include "blockingqueue.h"

#include "oclcontext.h"
#include "oclfilter.h"
#include "oclbayerfilter.h"
#include "oclmorphfilter.h"
#include "oclmorphhitmissfilter.h"

class oclProcThread : public QThread
{
	Q_OBJECT
public:
	oclProcThread(BlockingQueue<ProcessingItem>& queue);
	virtual ~ProcThread();

	void stop();
	virtual void run();

private:
	BlockingQueue<ProcessingItem>& queue;
	QMutex stopMutex;
	bool stopped;
	
	oclContext c;
	oclBayerFilter bayerFilter;
	oclMorphFilter morphFilter;
	oclMorphHitMissFilter hitmissFilter;

signals:
	void processingDone(const ProcessedItem& item);
};