#include "mainwindow.h"
#include "morphop.h"
#include "morphocl.h"

#include <QElapsedTimer>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>

#if !defined(_WIN32)
#include <sys/time.h>
#endif

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// Morph

MainWindow::MainWindow(QString filename, QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	// Menu
	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(openTriggered()));
	connect(ui.actionSave, SIGNAL(triggered()), this, SLOT(saveTriggered()));
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(exitTriggered()));
	connect(ui.actionOpenCL, SIGNAL(triggered(bool)), this, SLOT(openCLTriggered(bool)));

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
	connect(ui.rbThinning, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
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
	connect(ui.dialRotation, SIGNAL(valueChanged(int)), this, SLOT(rotationChanged(int)));
	connect(ui.pbResetRotation, SIGNAL(pressed()), this, SLOT(rotationResetPressed()));

#if 0
	ocl = new MorphOpenCLImage();
#else
	ocl = new MorphOpenCLBuffer();
#endif

	ocl->errorCallback = [this](const QString& message, cl_int err)
	{
		Q_UNUSED(err);
		QMessageBox::critical(this, "OpenCL error", message,
			QMessageBox::Ok);
		exit(1);
	};

	oclSupported = ocl->initOpenCL(CL_DEVICE_TYPE_CPU);
	if(oclSupported)
	{
		ui.actionOpenCL->setEnabled(true);
		ui.actionOpenCL->setChecked(true);
	}
	else
	{
		QMessageBox::critical(nullptr,
			"Critical error",
			"No OpenCL Platform available therefore OpenCL processing will be disabled",
			QMessageBox::Ok);

		ui.actionOpenCL->setEnabled(false);
		ui.actionOpenCL->setChecked(false);
	}

	openFile(filename);

	// Wartosci domyslne
	ui.rbNone->toggle();
	ui.rbRect->toggle();
	ui.cbSquare->setChecked(true);

	ui.lbXElementSize->setText(QString::fromLatin1("Horizontal: 3"));
	ui.lbYElementSize->setText(QString::fromLatin1("Vertical: 3"));
	ui.lbRotation->setText(QString::fromLatin1("0"));

	kradiusx = 1;
	kradiusy = 1;
	krotation = 0;

	statusBarLabel = new QLabel();
	ui.statusBar->addPermanentWidget(statusBarLabel);

	// TODO
	ui.rbVoronoi->setVisible(false);
	ui.sbPruning->setVisible(false);
	ui.cbPrune->setVisible(false);
}
// -------------------------------------------------------------------------
MainWindow::~MainWindow()
{

}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// Zdarzenia

void MainWindow::openTriggered()
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
void MainWindow::saveTriggered()
{
	QString filename = QFileDialog::getSaveFileName(this, QString(), ".",
		QString::fromLatin1("Image file (*.png)"));
	if(!filename.isEmpty())
	{
		ui.lbImage->pixmap()->toImage().save(filename);
	}
}
// -------------------------------------------------------------------------
void MainWindow::exitTriggered()
{
	close();
}
// -------------------------------------------------------------------------
void MainWindow::openCLTriggered(bool state)
{
	Q_UNUSED(state);
	refresh();
}
// -------------------------------------------------------------------------
void MainWindow::invertChanged(int state)
{
	Q_UNUSED(state);
	cv::Mat lut(1, 256, CV_8U);
	uchar* p = lut.ptr<uchar>();
	for(int i = 0; i < lut.cols; ++i)
	{
		*p++ = 255 - i;
	}
	cv::LUT(src, lut, src);

	if(oclSupported)
		ocl->setSourceImage(&src);

	refresh();
}
// -------------------------------------------------------------------------
void MainWindow::operationToggled(bool checked)
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
void MainWindow::structureElementToggled(bool checked)
{
	if(checked)
	{
		if(!ui.rbNone->isChecked())
			refresh();
	}
}
// -------------------------------------------------------------------------
void MainWindow::ratioChanged(int state)
{
	if(state == Qt::Checked)
	{
		int vv = qMax(ui.hsXElementSize->value(), ui.hsYElementSize->value());
		ui.hsXElementSize->setValue(vv);
		ui.hsYElementSize->setValue(vv);
	}
}
// -------------------------------------------------------------------------
void MainWindow::elementSizeXChanged(int value)
{
	Q_UNUSED(value);
	ui.lbXElementSize->setText(QString::fromLatin1("Horizontal: ") + 
		QString::number(2 * ui.hsXElementSize->value() + 1));

	if(ui.cbSquare->checkState() == Qt::Checked)
	{
		if(ui.hsYElementSize->value() != ui.hsXElementSize->value())
			ui.hsYElementSize->setValue(ui.hsXElementSize->value());
	}

	kradiusx = ui.hsXElementSize->value();

	if(!ui.rbNone->isChecked())
		refresh();
}
// -------------------------------------------------------------------------
void MainWindow::elementSizeYChanged(int value)
{
	Q_UNUSED(value);
	ui.lbYElementSize->setText(QString::fromLatin1("Vertical: ") + 
		QString::number(2 * ui.hsYElementSize->value() + 1));

	if(ui.cbSquare->checkState() == Qt::Checked)
	{
		if(ui.hsXElementSize->value() != ui.hsYElementSize->value())
			ui.hsXElementSize->setValue(ui.hsYElementSize->value());
	}

	kradiusy = ui.hsYElementSize->value();

	if(!ui.rbNone->isChecked())
		refresh();
}
// -------------------------------------------------------------------------
void MainWindow::rotationChanged(int value)
{
	Q_UNUSED(value);
	int angle = ui.dialRotation->value();
	if(angle >= 180) { angle -= 180; }
	else { angle += 180; }
	angle = 360 - angle;
	angle = angle % 360;

	ui.lbRotation->setText(QString::number(angle));
	krotation = angle;

	if(!ui.rbNone->isChecked())
		refresh();
}
// -------------------------------------------------------------------------
void MainWindow::rotationResetPressed()
{
	ui.dialRotation->setValue(180);
}
// -------------------------------------------------------------------------
void MainWindow::pruneChanged(int state)
{
	Q_UNUSED(state);
	if(ui.rbVoronoi->isChecked() && ui.sbPruning->value() != 0)
	{
		refresh();
	}
}

// Koniec zdarzen
// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH

void MainWindow::showCvImage(const cv::Mat& image)
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
void MainWindow::openFile(const QString& filename)
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
	if(oclSupported)
		ocl->setSourceImage(&src);

	this->resize(0, 0);
}
// -------------------------------------------------------------------------
void MainWindow::refresh()
{
	if(ui.rbNone->isChecked())
	{
		showCvImage(src);
		return;
	}

	// Operacje hit-miss
	if (ui.rbThinning->isChecked() ||
		ui.rbSkeleton->isChecked() || 
		ui.rbVoronoi->isChecked())
	{
		// deaktywuj wybor elementu strukturalnego
		ui.gbElement->setEnabled(false);
		ui.gbElementSize->setEnabled(false);
	}
	else
	{
		// deaktywuj wybor elementu strukturalnego
		ui.gbElement->setEnabled(true);
		ui.gbElementSize->setEnabled(true);
	}

	if(ui.actionOpenCL->isChecked())
		morphologyOpenCL();
	else
		morphologyOpenCV();
}
// -------------------------------------------------------------------------
void MainWindow::morphologyOpenCV()
{
#if defined(_WIN32)
	LARGE_INTEGER freq, start, end;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
#else
	timeval start, end;
	gettimeofday(&start, NULL);
#endif

	int iters = 1;
	EOperationType opType = operationType();
	cv::Mat dst;

	// Operacje hit-miss
	if (opType == OT_Thinning ||
		opType == OT_Skeleton || 
		opType == OT_Voronoi)
	{
		dst = src.clone();

		switch (opType)
		{
		case OT_Thinning:
			{
				morphologyThinning(src, dst);
				break;
			}

		case OT_Skeleton:
			{
				cv::Mat src1 = src.clone();
				iters = morphologySkeleton(src1, dst);

				// Szkielet - bialy
				// tlo - szare (zmiana z bialego)
				// obiekt - czarny
				dst = src/2 + dst;
				break;
			}

		case OT_Voronoi:
			{
				cv::Mat src1 = src.clone();
				iters = morphologyVoronoi(src1, dst,
					ui.cbPrune->isChecked() ? ui.sbPruning->value() : 0);

				// Strefy - szare
				// Reszta - niezmienione
				dst = dst/2 + src;		
			}
		default: break;
		}
	}
	else
	{
		int op_type;
		switch(opType)
		{
		case OT_Erode: op_type = cv::MORPH_ERODE; break;
		case OT_Dilate: op_type = cv::MORPH_DILATE; break;
		case OT_Open: op_type = cv::MORPH_OPEN; break;
		case OT_Close: op_type = cv::MORPH_CLOSE; break;
		case OT_Gradient: op_type = cv::MORPH_GRADIENT; break;
		case OT_TopHat: op_type = cv::MORPH_TOPHAT; break;
		case OT_BlackHat: op_type = cv::MORPH_BLACKHAT; break;
		default: op_type = cv::MORPH_ERODE; break;
		}

		cv::Mat element = standardStructuringElement();

		//if(opType == OT_Erode)
		//{			
		//	morphologyErode(src, dst, element);
		//}
		//else
		{
			cv::morphologyEx(src, dst, op_type, element);
		}
	}

	showCvImage(dst);

#if defined(_WIN32)
	QueryPerformanceCounter(&end);
	double elapsed = (static_cast<double>(end.QuadPart - start.QuadPart) / 
		static_cast<double>(freq.QuadPart)) * 1000.0f;
#else
	gettimeofday(&end, NULL);
	double elapsed = (static_cast<double>(end.tv_sec - start.tv_sec) * 1000 +
		0.001f * static_cast<double>(end.tv_usec - start.tv_usec));
#endif

	QString txt;
	QTextStream strm(&txt);
	strm << "Time elasped : " << /*timer.*/elapsed/*()*/ << " ms, iterations: " << iters;
	statusBarLabel->setText(txt);
}
// -------------------------------------------------------------------------
void MainWindow::morphologyOpenCL()
{
	cv::Mat dst, element = standardStructuringElement();
	EOperationType opType = operationType();

	int iters;
	ocl->setStructureElement(element);
	double delapsed = ocl->morphology(opType, dst, iters);
	
	// Wyswietl statystyki
	QString txt; 
	QTextStream strm(&txt);
	strm << "Time elasped : " << delapsed << " ms, iterations: " << iters;
	statusBarLabel->setText(txt);

	// Pokaz obraz wynikowy
	showCvImage(dst);
}
// -------------------------------------------------------------------------
cv::Mat MainWindow::standardStructuringElement()
{
	EStructureElementType type;

	if(ui.rbRect->isChecked()) type = SET_Rect;
	else if(ui.rbEllipse->isChecked()) type = SET_Ellipse;
	else if(ui.rbCross->isChecked()) type = SET_Cross;
	else type = SET_Diamond;

	return ::standardStructuringElement(
		kradiusx, kradiusy,
		type, krotation);
}
// -------------------------------------------------------------------------
EOperationType MainWindow::operationType()
{
	if(ui.rbErode->isChecked()) { return OT_Erode; }
	else if(ui.rbDilate->isChecked()) { return  OT_Dilate; }
	else if(ui.rbOpen->isChecked()) { return OT_Open; }
	else if(ui.rbClose->isChecked()) { return OT_Close; }
	else if(ui.rbGradient->isChecked()) { return OT_Gradient; }
	else if(ui.rbTopHat->isChecked()) { return OT_TopHat; }
	else if(ui.rbBlackHat->isChecked()) { return OT_BlackHat; }
	else if(ui.rbThinning->isChecked()) { return OT_Thinning; }
	else if(ui.rbSkeleton->isChecked()) { return OT_Skeleton; }
	else if(ui.rbVoronoi->isChecked()) { return OT_Voronoi; }
	else { return OT_Erode; }

}