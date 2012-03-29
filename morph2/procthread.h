#pragma once

#include <QThread>

#include "blockingqueue.h"

class ProcThread : public QThread
{
	Q_OBJECT
public:
	ProcThread(BlockingQueue<ProcessingItem>& queue);
	virtual ~ProcThread();

	void stop();
	virtual void run();

private:
	BlockingQueue<ProcessingItem>& queue;
	QMutex stopMutex;
	bool stopped;

signals:
	void processingDone(const ProcessedItem& item);
};