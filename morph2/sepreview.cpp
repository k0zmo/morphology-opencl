#include "sepreview.h"
#include "cvutils.h"
#include "controller.h"

#include <QImage>
#include <QMouseEvent>

PreviewLabel::PreviewLabel(QWidget* parent)
	: QLabel(parent)
{
	connect(this, SIGNAL(structuringElementModified(cv::Mat)), 
		gC, SLOT(onStructuringElementModified(cv::Mat)));
}

void PreviewLabel::setPreviewImage(const cv::Mat& se_)
{
	cv::Mat se(se_);
	this->se = se_;

	// Dostosuj go do wyswietlenia
	cv::Size previewSize(width(), height());
	cvu::fitImageToWholeSpace(se, previewSize);

	if(0)
	{
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;

		// Znajdz kontury
		cv::findContours(se, contours, hierarchy,
			CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

		// Znajdz otoczke wypukla dla kazdego konturu
		std::vector<std::vector<cv::Point> > hull(contours.size());
		for(size_t i = 0; i < contours.size(); ++i)
			cv::convexHull(cv::Mat(contours[i]), hull[i]);

		// Narysuj otoczke wypukla
		for(size_t i = 0; i < contours.size(); ++i)
			cv::drawContours(se, hull, i, cv::Scalar(2), 3, CV_AA);
	}

	// Konwersja cv::Mat -> QImage (QPixmap)
	QImage img(reinterpret_cast<const quint8*>(se.data),
		se.cols, se.rows, se.step, 
		QImage::Format_Indexed8);

	// Utworz palete 
	QVector<QRgb> colorTable;
	colorTable << qRgb(0, 0, 0) <<
				  qRgb(255, 255, 255) <<
				  qRgb(255, 255, 128); // <- dla konturow
	img.setColorTable(colorTable);

	setPixmap(QPixmap::fromImage(img));
}

void PreviewLabel::resizeEvent(QResizeEvent* evt)
{
	setPreviewImage(se);
	QLabel::resizeEvent(evt);
}

void PreviewLabel::mousePressEvent(QMouseEvent* evt)
{
	double dx = evt->x() / pixSize.width();
	double dy = evt->y() / pixSize.height();

	int x = static_cast<int>(dx);
	int y = static_cast<int>(dy);

	if(x >= se.cols || y >= se.rows)
		return;

	switch(evt->button())
	{
	case Qt::LeftButton:
		se.at<uchar>(y, x) = 1;
		break;
	case Qt::RightButton:
		se.at<uchar>(y, x) = 0;
		break;
	default: 
		return;
	}

	// poinformuj (controlera) o modyfikacji elementu strukturalnego
	emit structuringElementModified(se);

	setPreviewImage(se);
}

// _____________________________________________________________________________

StructuringElementPreview::StructuringElementPreview(QWidget* parent)
	: QDialog(parent)
{
	setupUi(this);
}

void StructuringElementPreview::onStructuringElementChanged(const cv::Mat& se_)
{
	previewLabel->setPreviewImage(se_);
}

void StructuringElementPreview::closeEvent(QCloseEvent* evt)
{
	Q_UNUSED(evt);
	emit closed();
}
