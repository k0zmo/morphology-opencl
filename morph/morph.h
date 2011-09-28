#ifndef MORPH_H
#define MORPH_H

#include <QtGui/QMainWindow>
#include "ui_morph.h"

#define CV_NO_BACKWARD_COMPATIBILITY

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class Morph : public QMainWindow
{
	Q_OBJECT

public:
	Morph(QString filename, QWidget *parent = 0, Qt::WFlags flags = 0);
	~Morph();

private:
	Ui::morphClass ui;
	QImage qsrc;
	cv::Mat src;

private:
	void showCvImage(const cv::Mat& image);
	void refresh();

	int morphologySkeleton(cv::Mat &src, cv::Mat &dst);
	int morphologyVoronoi(cv::Mat &src, cv::Mat &dst);

	void openFile(const QString& filename);

private slots:
	void openTriggered();
	void saveTriggered();
	void exitTriggered();

	void invertChanged(int state);
	void operationToggled(bool checked);
	void structureElementToggled(bool checked);

	void elementSizeXChanged(int value);
	void elementSizeYChanged(int value);
	void ratioChanged(int state);
	void rotationChanged(int value);
	void rotationResetPressed();

	void pruneChanged(int state);
};

#endif // MORPH_H
