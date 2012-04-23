#pragma once

#include "singleton.h"
#include "mainwidget.h"
#include "previewproxy.h"
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

#include "oclthread.h"
#include "ui_mainwindow.h"

enum EOpenCLMethod
{
	OM_Buffer1D,
	OM_Buffer2D
};

class Controller :
		public QMainWindow,
		public Singleton<Controller>,
		Ui::MainWindow
{
	Q_OBJECT
public:
	Controller(QWidget* parent = 0, Qt::WFlags flags = 0);
	virtual ~Controller();
	void start();

private:
	MainWidget* mw;
	Configuration conf;
	PreviewProxy* preview;
	GLDummyWidget* shareWidget;

	QLabel* procQueueLabel;
	QLabel* statusBarLabel;
	QLabel* cameraStatusLabel;

	bool negateSource;
	bool oclSupported;
	bool useOpenCL;
	bool autoTrigger;
	bool resizeCustomSe;
	bool cameraConnected;	

	BlockingQueue<ProcessingItem> procQueue;
	BlockingQueue<ProcessingItem> clQueue;

	ProcThread* procThread;
	CapThread* capThread;
	oclThread* clThread;

	cv::VideoCapture camera;
	cv::Mat src;
	cv::Mat dst;
	cv::Mat customSe;

private slots:
	void onPreviewInitialized(bool success);
	void onRecompute();
	void onProcessingDone(const ProcessedItem& item);

	// Menu (File)
	void onFromCameraTriggered(bool state);
	void onOpenFileTriggered();
	void onSaveFileTriggered();
	void onOpenStructuringElementTriggered();
	void onSaveStructuringElementTriggered();

	// Menu (Settings)
	void onOpenCLTriggered(bool state);
	void onPickMethodTriggerd();
	void onSettingsTriggered();

	void onInvertChanged(int state);
	void onAutoTriggerChanged(int state);
	void onBayerIndexChanged(int bcode);

	void onStructuringElementChanged();
	void onStructuringElementPreviewPressed();
	void onStructuringElementModified(const cv::Mat& customSe);

private:
	void openFile(const QString& filename);
	cv::Mat standardStructuringElement();
	cv::Mat structuringElement();
	void showStats(int iters, double elapsed);

	// Ustawia mozliwosc zapisu/odczytu obrazu do/z dysku
	// Uzywane przy aktywacji/deaktywacji wyjscia z kamery
	void setEnabledSaveOpenFile(bool state)
	{
		actionOpen->setEnabled(state);
		actionSave->setEnabled(state);
	}

	// Ustawia mozliwosc zaznaczenia "silnika" OpenCL
	// Uzywane w przypadku bledu inicjalizacji
	void setOpenCLCheckableAndChecked(bool state)
	{
		actionOpenCL->setEnabled(state);
		actionOpenCL->setChecked(state);
	}

	void setCameraStatusBarState(bool connected)
	{
		cameraStatusLabel->setText(connected ?
			"Camera: Connected" : "Camera: Not connected");
	}

	void setEnqueueJobsStatus()
	{
		if(useOpenCL)
			procQueueLabel->setText(QString("Enqueued jobs: %1").arg(clQueue.size()));
		else
			procQueueLabel->setText(QString("Enqueued jobs: %1").arg(procQueue.size()));
	}

	//
	// OpenCL
	//
private:
	void initializeOpenCL();
private slots:
	void onOpenCLInitialized(bool success);

signals:
	void structuringElementChanged(const cv::Mat& se);
};

#define gC Controller::getSingletonPtr()
