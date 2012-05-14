#pragma once

#include <QThread>

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "blockingqueue.h"

//#define SAPERA_SUPPORT

#ifdef SAPERA_SUPPORT
class SapAcquisition;
class SapBuffer;
class SapAcqToBuf;
#endif // SAPERA_SUPPORT

class CapThread : public QThread
{
	Q_OBJECT
public:
	CapThread(int usedQueue,
		BlockingQueue<ProcessingItem>& procQueue,
		BlockingQueue<ProcessingItem> &clQueue);

	bool openCamera(int camId);
#ifdef SAPERA_SUPPORT
	bool openCamera(const QString& ccf);
#endif // SAPERA_SUPPORT
	
	void closeCamera();
	void stop();

	virtual void run();

	void setNegateImage(bool negate);
	void setBayerCode(cvu::EBayerCode bc);
	void setMorpgologyOperation(cvu::EMorphOperation op);
	void setStructuringElement(const cv::Mat& se);
	
	void setJobDescription(bool negate, cvu::EBayerCode bc,
		cvu::EMorphOperation op, const cv::Mat& se);

	void setUsedQueue(int q);

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
	BlockingQueue<ProcessingItem>* queue[2];
	QMutex jobDescMutex;
	QMutex stopThreadMutex;
	ProcessingItem item;
	int usedQueue;

	int channels; // np. 3
	int depth; // np. CV_8U
	int width;
	int height;

	bool stopped;

	bool useSaperaLib;
#ifdef SAPERA_SUPPORT
	SapAcquisition* acq;
	SapBuffer* buffer;
	SapAcqToBuf* xfer;

	void freeSapera();
#endif // SAPERA_SUPPORT	
};
