#include "stdafx.h"
#include "morph.h"

#include <QElapsedTimer>
#include <QFileDialog>
#include <QMessageBox>

// -------------------------------------------------------------------------
cv::Mat QImage2cvMat(const QImage& qimage)
{
	cv::Mat mat = cv::Mat(
		qimage.height(),
		qimage.width(),
		CV_8UC4,
		const_cast<uchar*>(qimage.bits()),
		qimage.bytesPerLine()); 

	// Konwersja na RGB
	cv::Mat mat2 = cv::Mat(
		mat.rows,
		mat.cols,
		CV_8UC3); 

	int from_to[] = { 0,0,  1,1,  2,2 }; 
	cv::mixChannels(&mat, 1, &mat2, 1, from_to, 3); 
	return mat2; 
};
// -------------------------------------------------------------------------
QImage cvMat2QImage(const cv::Mat& image, QImage::Format format)
{
	QImage qimg(
		reinterpret_cast<const quint8*>(image.data),
		image.cols, image.rows, image.step, 
		format);

	if(format == QImage::Format_RGB888)
	{
		return qimg.rgbSwapped();
	}
	else /*if(format == QImage::Format_Indexed8)*/
	{
		return qimg;
	}	
};
// -------------------------------------------------------------------------
cv::Mat structuringElementDiamond(int radius)
{
	int a = radius;
	int s = 2 * radius + 1;

	cv::Mat element = cv::Mat(s, s, CV_8U, cv::Scalar(1));

	// top-left
	int y = a;
	for(int j = 0; j < a; ++j)
	{
		for(int i = 0; i < y; ++i)
		{
			element.at<uchar>(j, i) = 0;
		}
		--y;
	}


	// top-right
	y = a + 1;
	for(int j = 0; j < a; ++j)
	{
		for(int i = y; i < s; ++i)
		{
			element.at<uchar>(j, i) = 0;
		}
		++y;
	}

	// bottom-left
	y = 1;
	for(int j = a; j < s; ++j)
	{
		for(int i = 0; i < y; ++i)
		{
			element.at<uchar>(j, i) = 0;
		}
		++y;
	}

	// bottom-right
	y = s - 1;
	for(int j = a; j < s; ++j)
	{
		for(int i = y; i < s; ++i)
		{
			element.at<uchar>(j, i) = 0;
		}
		--y;
	}

	return element;
}
// -------------------------------------------------------------------------
int countDiffPixels(const cv::Mat& src1, const cv::Mat& src2)
{
	cv::Mat diff;
	cv::compare(src1, src2, diff, cv::CMP_NE);
	return cv::countNonZero(diff);
}

void morphRemove(const cv::Mat& src, cv::Mat& dst);
void morphSkeleton(cv::Mat src, cv::Mat dst);
void morphVoronoi(cv::Mat src, cv::Mat dst);
void morphPruning(cv::Mat src, cv::Mat dst);


// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// Morph

Morph::Morph(QString filename, QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	// Menu
	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(openTriggered()));
	connect(ui.actionSave, SIGNAL(triggered()), this, SLOT(saveTriggered()));
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(exitTriggered()));

	connect(ui.cbInvert, SIGNAL(stateChanged(int)), this, SLOT(invertChanged(int)));

	// Operacje
	connect(ui.rbNone, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbErode, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbDilate, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbOpen, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbClose, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbGradient, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbTopHat, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbBlackHat, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbRemove, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbSkeleton, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbVoronoi, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));

	//connect(ui.sbPruning, SIGNAL(valueChanged(int)), this, SLOT(pruningItersChanged(int)));
	connect(ui.sbPruning, SIGNAL(valueChanged(int)), this, SLOT(pruneChanged(int)));
	connect(ui.cbPrune, SIGNAL(stateChanged(int)), this, SLOT(pruneChanged(int)));

	// Element strukturalny
	connect(ui.rbRect, SIGNAL(toggled(bool)), this, SLOT(structureElementToggled(bool)));
	connect(ui.rbEllipse, SIGNAL(toggled(bool)), this, SLOT(structureElementToggled(bool)));
	connect(ui.rbCross, SIGNAL(toggled(bool)), this, SLOT(structureElementToggled(bool)));
	connect(ui.rbDiamond, SIGNAL(toggled(bool)), this, SLOT(structureElementToggled(bool)));

	// Rozmiar elementu strukturalnego
	connect(ui.cbSquare, SIGNAL(stateChanged(int)), this, SLOT(ratioChanged(int)));
	connect(ui.hsXElementSize, SIGNAL(valueChanged(int)), this, SLOT(elementSizeXChanged(int)));
	connect(ui.hsYElementSize, SIGNAL(valueChanged(int)), this, SLOT(elementSizeYChanged(int)));

	openFile(filename);

	// Wartosci domyslne
	ui.rbNone->toggle();
	ui.rbRect->toggle();
	ui.cbSquare->setChecked(true);

	ui.lbXElementSize->setText(QString::fromLatin1("Horizontal: 3"));
	ui.lbYElementSize->setText(QString::fromLatin1("Vertical: 3"));
}
// -------------------------------------------------------------------------
Morph::~Morph()
{

}
// -------------------------------------------------------------------------
void Morph::openTriggered()
{
	QString filename = QFileDialog::getOpenFileName(
		nullptr, QString(), ".",
		QString::fromLatin1("Image files (*.png *.jpg *.bmp)"));

	if(!filename.isEmpty())
	{
		openFile(filename);
		refresh();
	}
}
// -------------------------------------------------------------------------
void Morph::saveTriggered()
{
	QString filename = QFileDialog::getSaveFileName(this, QString(), ".",
		QString::fromLatin1("Image file (*.png)"));
	if(!filename.isEmpty())
	{
		ui.lbImage->pixmap()->toImage().save(filename);
	}
}
// -------------------------------------------------------------------------
void Morph::exitTriggered()
{
	close();
}
// -------------------------------------------------------------------------
void Morph::invertChanged(int state)
{
	cv::Mat lut(1, 256, CV_8U);
	uchar* p = lut.ptr<uchar>();
	for(int i = 0; i < lut.cols; ++i)
	{
		*p++ = 255 - i;
	}
	cv::LUT(src, lut, src);

	refresh();
}
// -------------------------------------------------------------------------
void Morph::operationToggled(bool checked)
{
	// Warunek musi byc spelniony bo sa zglaszane 2 zdarzenia
	// 1 - jeden z radiobuttonow zmienil stan z aktywnego na nieaktywny
	// 2 - zaznaczony radiobutton zmienil stan z nieaktywnego na aktywny
	if(checked)
	{
		refresh();
	}
}
// -------------------------------------------------------------------------
void Morph::structureElementToggled(bool checked)
{
	if(checked)
	{
		if(!ui.rbNone->isChecked())
			refresh();
	}
}
// -------------------------------------------------------------------------
void Morph::ratioChanged(int state)
{
	if(state == Qt::Checked)
	{
		int vv = qMax(ui.hsXElementSize->value(), ui.hsYElementSize->value());
		ui.hsXElementSize->setValue(vv);
		ui.hsYElementSize->setValue(vv);
	}
}
// -------------------------------------------------------------------------
void Morph::elementSizeXChanged(int value)
{
	ui.lbXElementSize->setText(QString::fromLatin1("Horizontal: ") + 
		QString::number(2 * ui.hsXElementSize->value() + 1));

	if(ui.cbSquare->checkState() == Qt::Checked)
	{
		if(ui.hsYElementSize->value() != ui.hsXElementSize->value())
			ui.hsYElementSize->setValue(ui.hsXElementSize->value());
	}

	if(!ui.rbNone->isChecked())
		refresh();
}
// -------------------------------------------------------------------------
void Morph::elementSizeYChanged(int value)
{
	ui.lbYElementSize->setText(QString::fromLatin1("Vertical: ") + 
		QString::number(2 * ui.hsYElementSize->value() + 1));

	if(ui.cbSquare->checkState() == Qt::Checked)
	{
		if(ui.hsXElementSize->value() != ui.hsYElementSize->value())
			ui.hsXElementSize->setValue(ui.hsYElementSize->value());
	}

	if(!ui.rbNone->isChecked())
		refresh();
}
// -------------------------------------------------------------------------
void Morph::pruneChanged(int state)
{
	if(ui.rbVoronoi->isChecked() && ui.sbPruning->value() != 0)
	{
		refresh();
	}
}
// -------------------------------------------------------------------------
void Morph::showCvImage(const cv::Mat& image, QImage::Format format)
{
	ui.lbImage->setPixmap(QPixmap::fromImage(cvMat2QImage(image, format)));
}
// -------------------------------------------------------------------------
void Morph::refresh()
{
	if(ui.rbNone->isChecked())
	{
		showCvImage(src, QImage::Format_RGB888);
		return;
	}

	cv::Point anchor(
		ui.hsXElementSize->value(),
		ui.hsYElementSize->value());

	cv::Size elem_size(
		2 * anchor.x + 1,
		2 * anchor.y + 1);

	QElapsedTimer timer;
	timer.start();

	int niters = 1;

	int op_type;
	if (ui.rbRemove->isChecked() ||
		ui.rbSkeleton->isChecked() || 
		ui.rbVoronoi->isChecked())
	{
		// deaktywuj wybor elementu strukturalnego
		ui.gbElement->setEnabled(false);
		ui.gbElementSize->setEnabled(false);

		// src1 - obraz jednokanalowy
		cv::Mat src1(src.size(), src.type());
		cvtColor(src, src1, CV_RGB2GRAY);
		cv::Mat dst = src1.clone();

		/*if(src1.elemSize() != sizeof(uchar))
		{
			QMessageBox::warning(this, 
				"Error", "Only binary images supported.", QMessageBox::Ok);
			return;
		}*/

		if(ui.rbRemove->isChecked())
		{
			morphRemove(src1, dst);
			showCvImage(dst, QImage::Format_Indexed8);
		}
		else if(ui.rbSkeleton->isChecked())
		{
			niters = morphologySkeleton(src1, dst);

			// Szkielet - bialy
			// tlo - szare (zmiana z bialego)
			// obiekt - czarny
			cv::Mat a;
			cvtColor(src, a, CV_RGB2GRAY);
			dst = a/2 + dst;

			showCvImage(dst, QImage::Format_Indexed8);
		}
		else if(ui.rbVoronoi->isChecked())
		{
			niters = morphologyVoronoi(src1, dst);

			// Strefy - czerwone
			// Reszta - niezmienione
			cv::Mat dd;
			cvtColor(dst, dd, CV_GRAY2BGR);

			std::vector<cv::Mat> planes;
			cv::split(dd, planes);

			std::for_each(
				planes[0].begin<uchar>(),
				planes[0].end<uchar>(),
				[](uchar& d) { d = 0; });

			std::for_each(
				planes[1].begin<uchar>(),
				planes[1].end<uchar>(),
				[](uchar& d) { d = 0; });

			cv::merge(planes, dd);
			dd = src + dd;

			showCvImage(dd, QImage::Format_RGB888);
		}
	}
	else
	{
		// aktywuj wybor elementu strukturalnego
		ui.gbElement->setEnabled(true);
		ui.gbElementSize->setEnabled(true);

		if(ui.rbErode->isChecked()) { op_type = cv::MORPH_ERODE; }
		else if(ui.rbDilate->isChecked()) { op_type = cv::MORPH_DILATE; }
		else if(ui.rbOpen->isChecked()) { op_type = cv::MORPH_OPEN; }
		else if(ui.rbClose->isChecked()) { op_type = cv::MORPH_CLOSE; }
		else if(ui.rbGradient->isChecked()) { op_type = cv::MORPH_GRADIENT; }
		else if(ui.rbTopHat->isChecked()) { op_type = cv::MORPH_TOPHAT; }
		else if(ui.rbBlackHat->isChecked()) { op_type = cv::MORPH_BLACKHAT; }

		cv::Mat element;

		if(ui.rbRect->isChecked())
		{
			element = cv::getStructuringElement(cv::MORPH_RECT, elem_size, anchor);
		}
		else if(ui.rbEllipse->isChecked())
		{
			element = cv::getStructuringElement(cv::MORPH_ELLIPSE, elem_size, anchor);
		}
		else if(ui.rbCross->isChecked())
		{
			element = cv::getStructuringElement(cv::MORPH_CROSS, elem_size, anchor);
		}
		else
		{
			element = structuringElementDiamond(qMin(anchor.x, anchor.y));
		}

		cv::Mat dst;
		cv::morphologyEx(src, dst, op_type, element);
		showCvImage(dst, QImage::Format_RGB888);
	}

	QString txt;
	QTextStream strm(&txt);
	strm << "Time elasped : " << timer.elapsed() << " ms, iterations: " << niters;

	ui.statusBar->showMessage(txt);
}
// -------------------------------------------------------------------------
void Morph::openFile(const QString& filename)
{
	qsrc = QImage(filename);
	if(qsrc.format() != QImage::Format_RGB32)
		qsrc = qsrc.convertToFormat(QImage::Format_RGB32);

	src = QImage2cvMat(qsrc);
	this->resize(0, 0);
}
// -------------------------------------------------------------------------
int Morph::morphologySkeleton(cv::Mat &src, cv::Mat &dst) 
{
	int niters = 0;

	while(true) 
	{
		// iteracja
		morphSkeleton(src, dst);
		++niters;

		// warunek stopu
		if(countDiffPixels(src, dst) == 0) break;

		src = dst.clone();
	}

	return niters;
}
// -------------------------------------------------------------------------
int Morph::morphologyVoronoi(cv::Mat &src, cv::Mat &dst) 
{
	int niters = 0;

	src = 255 - src;
	dst = 255 - dst;

	while(true) 
	{
		// iteracja
		//morphVoronoi(src, dst);
		morphSkeleton(src, dst);
		++niters;

		// warunek stopu
		if(countDiffPixels(src, dst) == 0) break;

		src = dst.clone();
	}

	if(ui.cbPrune->isChecked())
	{
		for(int i = 0; i < ui.sbPruning->value(); ++i)
		{
			src = dst.clone();

			// iteracja
			morphPruning(src, dst);
			++niters;				
		}
	}

	return niters;
}
