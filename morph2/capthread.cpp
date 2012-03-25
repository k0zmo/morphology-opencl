#include "capthread.h"

#include <QDebug>

CapThread::CapThread( BlockingQueue<ProcessingItem>& queue)
	: QThread(nullptr)
	, queue(queue)
	, channels(0)
	, depth(0)
	, width(0) 
	, height(0)
{
	item.op = Morphology::OT_None;
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

void CapThread::run()
{
	while(true)
	{
		camera >> item.src;

		// TODO: hardcoded
		if(item.src.channels() != 1)
		{
			int code = (item.src.channels() == 3) ? 
				CV_BGR2GRAY : CV_BGRA2GRAY;
			cvtColor(item.src, item.src, code);
		}

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

void CapThread::setMorpgologyOperation(Morphology::EOperationType op)
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
	Morphology::EOperationType op, const cv::Mat& se)
{
	QMutexLocker locker(&jobDescMutex);
	item.negate = negate;
	item.bc = bc;
	item.op = op;
	item.se = se;
}