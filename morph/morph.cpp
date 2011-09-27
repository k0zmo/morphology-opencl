#include "morph.h"

#include <QElapsedTimer>
#include <QFileDialog>
#include <QMessageBox>

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
void Morph::showCvImage(const cv::Mat& image)
{
	auto toQImage = [](const cv::Mat& image)
	{
		return QImage(
			reinterpret_cast<const quint8*>(image.data),
			image.cols, image.rows, image.step, 
			QImage::Format_Indexed8);
	};

	ui.lbImage->setPixmap(QPixmap::fromImage(toQImage(image)));
}
// -------------------------------------------------------------------------
void Morph::refresh()
{
	if(ui.rbNone->isChecked())
	{
		showCvImage(src);
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

		cv::Mat dst = src.clone();

		if(ui.rbRemove->isChecked())
		{
			morphRemove(src, dst);
			showCvImage(dst);
		}
		else if(ui.rbSkeleton->isChecked())
		{
			cv::Mat src1 = src.clone();
			niters = morphologySkeleton(src1, dst);

			// Szkielet - bialy
			// tlo - szare (zmiana z bialego)
			// obiekt - czarny
			dst = src/2 + dst;

			showCvImage(dst);
		}
		else if(ui.rbVoronoi->isChecked())
		{
			cv::Mat src1 = src.clone();
			niters = morphologyVoronoi(src1, dst);

			// Strefy - szare
			// Reszta - niezmienione
			dst = dst/2 + src;

			showCvImage(dst);
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
		showCvImage(dst);
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
	if(qsrc.format() != QImage::Format_RGB888)
		qsrc = qsrc.convertToFormat(QImage::Format_RGB888);

	auto toCvMat = [](const QImage& qimage) -> cv::Mat
	{
		cv::Mat mat(qimage.height(), qimage.width(), CV_8UC3,
			const_cast<uchar*>(qimage.bits()),
			qimage.bytesPerLine());

		// Konwersja do obrazu jednokanalowego
		cvtColor(mat, mat, CV_RGB2GRAY);
		return mat;
	};

	src = toCvMat(qsrc);
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

	// Diagram voronoi jest operacja dualna do szkieletowania
	src = 255 - src;
	dst = 255 - dst;

	while(true) 
	{
		// iteracja
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
