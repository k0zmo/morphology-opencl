#include "capthread.h"

#include <QDebug>

#ifdef SAPERA_SUPPORT
#	include "SapClassBasic.h"
#	ifdef _DEBUG
#		pragma comment(lib, "SapClassBasicD.lib")
#	else
#		pragma comment(lib, "SapClassBasic.lib")
#	endif
#endif

CapThread::CapThread(int usedQueue,
	BlockingQueue<ProcessingItem>& procQueue,
	BlockingQueue<ProcessingItem>& clQueue)
	: QThread(nullptr)
	, usedQueue(usedQueue)
	, channels(0)
	, depth(0)
	, width(0) 
	, height(0)
	, stopped(false)  
	, useSaperaLib(false)	
{
	item.op = cvu::MO_None;
	item.bc = cvu::BC_None;
	item.negate = false;
	item.se = cv::Mat();

	queue[0] = &procQueue;
	queue[1] = &clQueue;
}

bool CapThread::openCamera(int camId)
{
	if(camera.open(camId))
	{
		width = camera.get(CV_CAP_PROP_FRAME_WIDTH);
		height = camera.get(CV_CAP_PROP_FRAME_HEIGHT);
		
		useSaperaLib = false;

		// CV_CAP_PROP_FORMAT zwraca tylko format danych (np. CV_8U),
		// bez liczby kanalow
		cv::Mat dummy;
		camera.read(dummy);
		channels = dummy.channels();
		depth = dummy.depth();

		qDebug() << "Frame parameters:" << 
			width << "x" << 
			height << "x" << 
			channels << "channels" <<
			"(format:" << QString(cvu::cvFormatToString(depth)) + ")"; 

		return true;
	}
	else
	{
		return false;
	}
}

#ifdef SAPERA_SUPPORT
bool CapThread::openCamera(const QString& ccf)
{
	SapLocation loc("X64-CL_iPro_1", 0);
	const char* cfg_file = ccf.toLatin1().constData();
	acq = new SapAcquisition(loc, cfg_file);
	buffer = new SapBuffer(1, acq);
	xfer = new SapAcqToBuf(acq, buffer);
	
	useSaperaLib = true;

	if(acq && !acq->Create())
	{
		printf("Create Acq error.\n");
		freeSapera();
		return false;
	}

	// Create buffer object
	if(buffer && !buffer->Create())
	{
		printf("Create Buffers error.\n");
		freeSapera();
		return false;
	}

	// Create transfer object
	if (xfer && !xfer->Create()) {
		printf("Create Xfer error.\n");
		freeSapera();
		return false;
	}

	image = cvCreateImage
		(cvSize(buffer->GetWidth(), 
		 buffer->GetHeight()), 
		 IPL_DEPTH_16U, 1);
	qDebug() << "Buffer size:" << buffer->GetWidth() << buffer->GetHeight();
	qDebug() << buffer->GetPixelDepth(); // ile bitow na pixel
	qDebug() << buffer->GetFormat();

	return true;
}

void CapThread::freeSapera()
{
	// Destroy transfer object
	if (xfer && !xfer->Destroy()) 
		return;

	// Destroy buffer object
	if (buffer && !buffer->Destroy())
		return;

	// Destroy acquisition object
	if (acq && !acq->Destroy()) 
		return;

	// Delete all objects
	if (xfer) delete xfer;
	if (buffer) delete buffer; 
	if (acq) delete acq; 

	image->imageData = nullptr;
	cvReleaseImage(&image);
	image = 0;

	xfer = 0;
	buffer = 0;
	acq = 0;
}
#endif // SAPERA_SUPPORT

void CapThread::closeCamera()
{
	if(!useSaperaLib)
	{
		if(camera.isOpened())
			camera.release();
	}
#ifdef SAPERA_SUPPORT
	else
	{
		if(!xfer)
		{
			printf("xfer is nullptr");
			return;
		}

		xfer->Freeze();
		if (!xfer->Wait(5000))
			printf("Grab could not stop properly.\n");

		freeSapera();
	}
#endif // SAPERA_SUPPORT
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
			{
				closeCamera();
 				break;
			}
		}

		if(!useSaperaLib)
		{
			camera >> frame;
		}
#ifdef SAPERA_SUPPORT
		else
		{
			xfer->Snap();
			xfer->Wait(1000);
			void* data;
			buffer->GetAddress(&data);
			image->imageData = static_cast<char*>(data);
			frame = cv::Mat(image);
		}
#endif
		//int prechannel = frame.channels();
		//int predepth = frame.depth();

		if(frame.depth() != CV_8U)
		{
			cv::Mat tmp = frame.clone();
			//tmp.convertTo(frame, CV_8UC1, 0.00625);
			tmp.convertTo(frame, CV_8UC1, 0.0625);
		}

		// Nie wiem czy to do konca jest poprawne
		if(frame.channels() != 1)
		{
			int code = (frame.channels() == 3)
				? CV_BGR2GRAY
				: CV_BGRA2GRAY;
			cvtColor(frame, frame, code);
		}
		
		//qDebug() << "\t\t\t\t\t" << frame.channels() << 
		//	prechannel << frame.depth() << predepth;		

		item.src = frame;

		{
			QMutexLocker locker(&jobDescMutex);
			queue[usedQueue]->enqueue(item);
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

void CapThread::setUsedQueue(int q)
{
	if(q != 0 && q != 1)
		return;

	QMutexLocker locker(&jobDescMutex);
	if(!queue[usedQueue]->isEmpty())
		queue[usedQueue]->clear();

	usedQueue = q;
}

cv::Mat CapThread::currentFrame() const
{
	return frame;
}
