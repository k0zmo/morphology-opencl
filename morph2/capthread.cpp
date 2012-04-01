#include "capthread.h"

#include <QDebug>

CapThread::CapThread( BlockingQueue<ProcessingItem>& queue)
	: QThread(nullptr)
	, queue(queue)
	, channels(0)
	, depth(0)
	, width(0) 
	, height(0)
	, stopped(false)
{
	item.op = cvu::MO_None;
	item.bc = cvu::BC_None;
	item.negate = false;
	item.se = cv::Mat();
}

bool CapThread::openCamera(int camId)
{
	if(camera.open(camId))
	{
		width = camera.get(CV_CAP_PROP_FRAME_WIDTH);
		height = camera.get(CV_CAP_PROP_FRAME_HEIGHT);

		// CV_CAP_PROP_FORMAT zwraca tylko format danych (np. CV_8U),
		// bez liczby kanalow
		cv::Mat dummy;
		camera.read(dummy);
		channels = dummy.channels();
		depth = dummy.depth();

		qDebug() << width << "x" << height << "x" << channels << depth;

		return true;
	}
	else
	{
		return false;
	}
}

void CapThread::closeCamera()
{
	if(camera.isOpened())
		camera.release();
}

void CapThread::stop()
{
	QMutexLocker locker(&stopThreadMutex);
	stopped = true;
}

void CapThread::run()
{
	while(true)
	{
		{
			QMutexLocker locker(&stopThreadMutex);
			if(stopped)
				break;
		}

		camera >> frame;

		// Nie wiem czy to do konca jest poprawne
		if(frame.channels() != 1)
		{
			int code = (frame.channels() == 3) ? 
				CV_BGR2GRAY : CV_BGRA2GRAY;
			cvtColor(frame, frame, code);
		}

		item.src = frame;

		{
			QMutexLocker locker(&jobDescMutex);
			queue.enqueue(item);
		}		
	}
}

void CapThread::setNegateImage(bool negate)
{
	QMutexLocker locker(&jobDescMutex);
	item.negate = negate;
}

void CapThread::setBayerCode(cvu::EBayerCode bc)
{
	QMutexLocker locker(&jobDescMutex);
	item.bc = bc;
}

void CapThread::setMorpgologyOperation(cvu::EMorphOperation op)
{
	QMutexLocker locker(&jobDescMutex);
	item.op = op;
}

void CapThread::setStructuringElement(const cv::Mat& se)
{
	QMutexLocker locker(&jobDescMutex);
	item.se = se;
}

void CapThread::setJobDescription(bool negate, cvu::EBayerCode bc,
	cvu::EMorphOperation op, const cv::Mat& se)
{
	QMutexLocker locker(&jobDescMutex);
	item.negate = negate;
	item.bc = bc;
	item.op = op;
	item.se = se;
}

cv::Mat CapThread::currentFrame() const
{
	return frame;
}