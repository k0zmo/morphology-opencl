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
	void stop();

	virtual void run();

	void setNegateImage(bool negate);
	void setBayerCode(cvu::EBayerCode bc);
	void setMorpgologyOperation(cvu::EMorphOperation op);
	void setStructuringElement(const cv::Mat& se);
	
	void setJobDescription(bool negate, cvu::EBayerCode bc,
		cvu::EMorphOperation op, const cv::Mat& se);

	int frameChannels() const
	{ return channels; }
	int frameDepth() const 
	{ return depth; }
	int frameHeight() const 
	{ return height; }
	int frameWidth() const 
	{ return width; }

	cv::Mat currentFrame() const;

private:
	cv::VideoCapture camera;
	cv::Mat frame;
	BlockingQueue<ProcessingItem>& queue;
	QMutex jobDescMutex;
	QMutex stopThreadMutex;
	ProcessingItem item;

	int channels; // np. 3
	int depth; // np. CV_8U
	int width;
	int height;

	bool stopped;
};