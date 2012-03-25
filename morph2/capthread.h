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
	CapThread(BlockingQueue<ProcessingItem>& queue);

	bool openCamera(int camId);
	void closeCamera();

	virtual void run();

	void setNegateImage(bool negate);
	void setBayerCode(cvu::EBayerCode bc);
	void setMorpgologyOperation(Morphology::EOperationType op);
	void setStructuringElement(const cv::Mat& se);
	
	void setJobDescription(bool negate, cvu::EBayerCode bc,
		Morphology::EOperationType op, const cv::Mat& se);

	int frameChannels() const
	{ return channels; }
	int frameDepth() const 
	{ return depth; }
	int frameHeight() const 
	{ return height; }
	int frameWidth() const 
	{ return width; }

private:
	cv::VideoCapture camera;
	BlockingQueue<ProcessingItem>& queue;
	QMutex jobDescMutex;
	ProcessingItem item;

	int channels; // np. 3
	int depth; // np. CV_8U
	int width;
	int height;
};