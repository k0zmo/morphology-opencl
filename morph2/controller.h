#pragma once

#include "singleton.h"
#include "mainwindow.h"
#include "glwidget.h"

#include "cvutils.h"
#include "morphop.h"

#include "configuration.h"
#include "blockingqueue.h"

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class CapThread;
class ProcThread;

enum EOpenCLMethod
{
	OM_Buffer1D,
	OM_Buffer2D
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
	//MorphOpenCL* ocl;

	bool negateSource;
	bool oclSupported;
	bool useOpenCL;
	bool autoTrigger;
	bool resizeCustomSe;
	bool cameraConnected;	

	BlockingQueue<ProcessingItem> procQueue;
	ProcThread* procThread;
	CapThread* capThread;

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

	void onRecompute();
	void onProcessingDone(const ProcessedItem& item);

private:
	void openFile(const QString& filename);
	cv::Mat standardStructuringElement();
	cv::Mat structuringElement();
	void showStats(int iters, double elapsed);
	void previewCpuImage(const cv::Mat& image);

	void initializeOpenCL(EOpenCLMethod method);
	//void setOpenCLSourceImage();
	//void processOpenCL(cvu::EOperationType op, const cv::Mat& se);
	//void previewGpuImage();
signals:
	void structuringElementChanged(const cv::Mat& se);
};

#define gC Controller::getSingletonPtr()