#pragma once

#include <QThread>

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "blockingqueue.h"

class CapThread : public QThread
{
	Q_OBJECT
public:
	CapThread(int camId, BlockingQueue<ProcessingItem>& queue)
		: QThread(nullptr)
		, queue(queue)
		, camId(camId)
		, format(0)
		, width(0) 
		, height(0)
	{
	}

	virtual void run()
	{
		camera.open(camId);

		width = camera.get(CV_CAP_PROP_FRAME_WIDTH);
		height = camera.get(CV_CAP_PROP_FRAME_HEIGHT);

		// CV_CAP_PROP_FORMAT zwraca tylko format danych (np. CV_8U),
		// bez liczby kanalow
		cv::Mat dummy;
		camera.read(dummy);
		int type = dummy.type();

		while(true)
		{
			camera >> item.src;

			//	Morphology::EOperationType op;
				//cvu::EBayerCode bc;
			//	cv::Mat se;
				//cv::Mat src;
			queue.tryEnqueue(item);
		}
	}

	//void setStructuringElement(const cv::Mat se);
	//void setBayerCode(cvu::EBayerCode bc);
	//void setMorpgologyOperation(Morphology::EOperationType op);

	bool isCameraConnected() const 
	{ return camera.isOpened(); }
	int frameFormat() const 
	{ return format; }
	int frameHeight() const 
	{ return height; }
	int frameWidth() const 
	{ return width; }

private:
	cv::VideoCapture camera;
	cv::Mat frame;
	BlockingQueue<ProcessingItem>& queue;

	int camId;
	int format;
	int width;
	int height;
	ProcessingItem item;
};