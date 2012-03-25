#pragma once

#include <QThread>

#include "blockingqueue.h"

class ProcThread : public QThread
{
	Q_OBJECT
public:
	ProcThread(BlockingQueue<ProcessingItem>& queue);
	virtual ~ProcThread();

	virtual void run();

private:
	BlockingQueue<ProcessingItem>& queue;

signals:
	void processingDone(const ProcessedItem& item);
};