#pragma once

#include <QtGui/QMainWindow>
#include "ui_mainwindow.h"

#include "morphocl.h"

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class MainWindow : public QMainWindow
{
	Q_OBJECT
public:
	MainWindow(QString filename, QWidget *parent = 0,
		Qt::WFlags flags = 0);
	virtual ~MainWindow();

private:
	Ui::mainWindowClass ui;
	QLabel* statusBarLabel;
	cv::Mat src, dst;
	cv::VideoCapture camera;

	bool disableRefreshing;
	int krotation;
	int timerId;
	int maxImageWidth;
	int maxImageHeight;

	MorphOpenCL* ocl;
	bool oclSupported;
	int method;

private:
	// Inicjalizuje OpenCLa
	void initOpenCL(int method);
	
	// Ustawia podany obraz w oknie podgladu
	void showCvImage(const cv::Mat& image);
	void showGlImage(int w, int h);
	
	// Czyta plik graficzny o podanej nazwie
	void openFile(const QString& filename);
	
	// Zwraca macierz elementu strukturalnego zgodnego 
	// z obecnymi ustawieniami
	cv::Mat standardStructuringElement();

	// Zwraca obecnie aktywna operacje morfologiczna
	Morphology::EOperationType operationType();

	// Odswieza widok, jesli trzeba wykona od nowa wskazana operacje
	void refresh();

	// Wykonuje operacje morfologiczne z uzyciem biblioteki OpenCV
	void morphologyOpenCV();

	// Wykonuje operacje morfologiczne z uzyciem API OpenCL'a
	void morphologyOpenCL();

	void timerEvent(QTimerEvent* event);

	void setOpenCLSourceImage();;

private slots:
	void openTriggered();
	void saveTriggered();
	void exitTriggered();
	void openCLTriggered(bool state);
	void pickMethodTriggered();
	void settingsTriggered();
	void cameraInputTriggered(bool state);

	void openSETriggered();
	void saveSETriggered();

	void invertChanged(int state);
	void bayerIndexChanged(int i);
	void noneOperationToggled(bool checked);
	void operationToggled(bool checked);
	void structuringElementToggled(bool checked);
	void structuringElementPreview();

	void ratioChanged(int state);
	void elementSizeXChanged(int value);
	void elementSizeYChanged(int value);
	void rotationChanged(int value);
	void rotationResetPressed();

	void runPressed();
	void autoRunChanged(int state);
};
