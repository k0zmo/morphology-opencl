#pragma once

#include <QQueue>
#include <QMutex>
#include <QWaitCondition>

#include "cvutils.h"
#include "morphop.h"

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
		QMutexLocker locker(&mutex);

		while(queue.size() >= maxSize)
			fullQueueWait.wait(&mutex);

		queue.enqueue(item);
		emptyQueueWait.wakeOne();
	}

	bool tryEnqueue(const T& item)
	{
		QMutexLocker locker(&mutex);

		if(queue.size() >= maxSize)
			return false;

		queue.enqueue(item);
		emptyQueueWait.wakeOne();
		return true;
	}

	T dequeue()
	{
		QMutexLocker locker(&mutex);

		while(queue.isEmpty())
			emptyQueueWait.wait(&mutex);

		T ret = queue.dequeue();
		fullQueueWait.wakeOne();
		return ret;
	}

	void clear()
	{
		QMutexLocker locker(&mutex);

		queue.clear();
		fullQueueWait.wakeOne();
	}

	bool isEmpty()
	{
		QMutexLocker locker(&mutex);
		return queue.isEmpty();
	}

	size_t size()
	{
		QMutexLocker locker(&mutex);
		return queue.size();
	}

private:
	int maxSize;
	QQueue<T> queue;
	QMutex mutex;
	QWaitCondition emptyQueueWait;
	QWaitCondition fullQueueWait;
};

struct ProcessingItem
{
	bool negate;
	cvu::EBayerCode bc;
	cvu::EMorphOperation op;
	cv::Mat se;
	cv::Mat src;
};

struct ProcessedItem
{
	int iters;
	double delapsed;
	cv::Mat dst;
};
