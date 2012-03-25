#pragma once

#include <QQueue>
#include <QMutex>
#include <QWaitCondition>

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "morphoperators.h"
#include "cvutils.h"

template<class T>
class BlockingQueue
{
public:
	BlockingQueue(int maxSize)
		: maxSize(maxSize)
	{
	}

	void enqueue(const T& item)
	{
		QMutexLocker locker(&enqueueMutex);

		while(queue.size() >= maxSize)
			fullQueueWait.wait(&enqueueMutex);

		queue.enqueue(item);
		emptyQueueWait.wakeOne();
	}

	bool tryEnqueue(const T& item)
	{
		QMutexLocker locker(&enqueueMutex);

		if(queue.size() >= maxSize)
			return false;

		queue.enqueue(item);
		emptyQueueWait.wakeOne();
		return true;
	}

	T dequeue()
	{
		QMutexLocker locker(&dequeueMutex);

		while(queue.isEmpty())
			emptyQueueWait.wait(&dequeueMutex);

		T ret = queue.dequeue();
		fullQueueWait.wakeOne();
		return ret;
	}

	void clear()
	{
		QMutexLocker locker(&dequeueMutex);
		QMutexLocker locker2(&enqueueMutex);

		queue.clear();
		fullQueueWait.wakeOne();
	}

private:
	int maxSize;
	QQueue<T> queue;
	QMutex enqueueMutex;
	QMutex dequeueMutex;
	QWaitCondition emptyQueueWait;
	QWaitCondition fullQueueWait;
};

struct ProcessingItem
{
	bool negate;
	Morphology::EOperationType op;
	cvu::EBayerCode bc;
	cv::Mat se;
	cv::Mat src;
};

struct ProcessedItem
{
	int iters;
	double delapsed;
	cv::Mat dst;
};
