#include "sepreview.h"
#include "cvutils.h"

#include <QImage>

SEPreview::SEPreview(QWidget* parent)
	: QDialog(parent)
{
	setupUi(this);
}

void SEPreview::onStructuringElementChanged(const cv::Mat& se_)
{
	cv::Mat se(se_);

	// Dostosuj go do wyswietlenia
	cv::Size previewSize(lbSEPreview->width(), lbSEPreview->height());
	CvUtil::resizeWithAspect(se, previewSize);

	// Konwersja cv::Mat -> QImage (QPixmap)
	QImage img(reinterpret_cast<const quint8*>(se.data),
		se.cols, se.rows, se.step, 
		QImage::Format_Indexed8);

	// Utworz palete 
	QVector<QRgb> colorTable;
	colorTable << qRgb(0, 0, 0) << qRgb(255, 255, 255);
	img.setColorTable(colorTable);

	lbSEPreview->setPixmap(QPixmap::fromImage(img));
}