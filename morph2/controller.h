#pragma once

#include "mainwindow.h"
#include "glwidget.h"

#include "morphocl.h"
#include "morphoclbuffer.h"
#include "morphoclimage.h"
#include "morphoperators.h"
#include "configuration.h"
#include "procthread.h"

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

template <typename T>
class Singleton
{
private:
	Singleton(const Singleton<T>&);
	Singleton& operator=(const Singleton<T>&);

protected:
	static T* msSingleton;

public:
	Singleton()
	{ Q_ASSERT(!msSingleton); msSingleton = static_cast<T*>(this); }

	~Singleton()
	{ Q_ASSERT(msSingleton); msSingleton = 0; }

	static T& getSingleton()
	{ Q_ASSERT(msSingleton); return *msSingleton; }

	static T* getSingletonPtr()
	{ return msSingleton; }
};

class Controller : public QObject, public Singleton<Controller>
{
	Q_OBJECT
public:
	Controller();
	virtual ~Controller();

	void start();

private:
	MainWindow* mw;
	Configuration conf;

	QLabel* previewLabel;
	GLWidget* previewWidget;
	MorphOpenCL* ocl;

	BlockingQueue<ProcessingItem> procQueue;
	ProcThread procThread;

	bool negateSource;
	bool oclSupported;
	bool useOpenCL;
	bool autoTrigger;
	bool resizeCustomSe;

	cv::VideoCapture camera;
	cv::Mat src;
	cv::Mat dst;
	cv::Mat customSe;

private slots:
	void onFromCameraTriggered(bool state);
	void onOpenFileTriggered();
	void onSaveFileTriggered();
	void onOpenStructuringElementTriggered();
	void onSaveStructuringElementTriggered();

	void onOpenCLTriggered(bool state);
	void onPickMethodTriggerd();
	void onSettingsTriggered();

	void onInvertChanged(int state);
	void onAutoTriggerChanged(int state);
	void onBayerIndexChanged(int bcode);

	void onStructuringElementChanged();
	void onStructuringElementPreviewPressed();
	void onStructuringElementModified(const cv::Mat& customSe);

	void onShowSourceImage();
	void onRecompute();

	void onProcessingDone(const ProcessedItem& item);

private:
	void openFile(const QString& filename);
	cv::Mat standardStructuringElement();
	void showStats(int iters, double elapsed);

	void initializeOpenCL(EOpenCLMethod method);
	void setOpenCLSourceImage();

	void processOpenCL(Morphology::EOperationType op, const cv::Mat& se);
	void processOpenCV(Morphology::EOperationType op, const cv::Mat& se);

	void previewCpuImage(const cv::Mat& image);
	void previewGpuImage();

signals:
	void structuringElementChanged(const cv::Mat& se);
};

#define gC Controller::getSingletonPtr()