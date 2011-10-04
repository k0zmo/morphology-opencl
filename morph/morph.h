#pragma once

#include <QtGui/QMainWindow>
#include "ui_morph.h"

#define CV_NO_BACKWARD_COMPATIBILITY

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <CL/cl.hpp>

class Morph : public QMainWindow
{
	Q_OBJECT

public:
	Morph(QString filename, QWidget *parent = 0, Qt::WFlags flags = 0);
	~Morph();

private:
	Ui::morphClass ui;
	QLabel* statusBarLabel;
	QImage qsrc;
	cv::Mat src;

	cl::Context context;
	cl::Device dev;
	cl::CommandQueue cq;

	cl::Kernel kernelSubtract;
	cl::Kernel kernelAddHalf;

	cl::Kernel kernelErode;
	cl::Kernel kernelDilate;
	cl::Kernel kernelRemove;

	cl::Kernel kernelSkeleton_iter[8];

	cl::Buffer clSrc;
	cl::Buffer clDst;
	cl::Buffer clElement;

	cl::Buffer clTmp;
	cl::Buffer clTmp2;

private:
	// Ustawia podany obraz w oknie podgladu
	void showCvImage(const cv::Mat& image);
	
	// Czyta plik graficzny o podanej nazwie
	void openFile(const QString& filename);
	
	// Zwraca macierz elementu strukturalnego zgodnego 
	// z obecnymi ustawieniami
	cv::Mat standardStructuringElement();

	// Odswieza widok, jesli trzeba wykona od nowa wskazana operacje
	void refresh();
	// Wykonuje operacje morfologiczne z uzyciem biblioteki OpenCV
	void morphologyOpenCV();

	// Inicjalizuje OpenCL'a
	void initOpenCL();
	// Pomocznicza funkcja do zglaszania bledow OpenCL'a
	void clError(const QString& message, cl_int err);

	// Wykonuje operacje morfologiczne z uzyciem API OpenCL'a
	void morphologyOpenCL();

	cl_ulong elapsedEvent(const cl::Event& evt);
	
	// Pomocnicza funkcja do odpalania kerneli do podst. operacji morfologicznych
	cl_ulong executeMorphologyKernel(cl::Kernel* kernel, const cl::Buffer& clBufferSrc,
		const cl::Buffer& clBufferDst);
	// Pomocnicza funkcja do odpalania kernela do odejmowania dwoch obrazow od siebie
	cl_ulong executeSubtractKernel(const cl::Buffer& clBufferA, const cl::Buffer& clBufferB,
		const cl::Buffer& clBufferDst);

private slots:
	void openTriggered();
	void saveTriggered();
	void exitTriggered();
	void openCLTriggered(bool state);

	void invertChanged(int state);
	void operationToggled(bool checked);
	void structureElementToggled(bool checked);

	void ratioChanged(int state);
	void elementSizeXChanged(int value);
	void elementSizeYChanged(int value);
	void rotationChanged(int value);
	void rotationResetPressed();

	void pruneChanged(int state);
};
