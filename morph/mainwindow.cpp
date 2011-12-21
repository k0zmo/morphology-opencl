#include "mainwindow.h"

#include "morphoclimage.h"
#include "morphoclbuffer.h"

#include <QElapsedTimer>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>
#include <QSettings>

#if !defined(_WIN32)
#include <sys/time.h>
#endif

#include "ui_sepreview.h"

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// Morph

MainWindow::MainWindow(QString filename, QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags),
	disableRefreshing(false),
	kradiusx(1),
	kradiusy(1),
	krotation(0)
{
	ui.setupUi(this);

	// Menu
	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(openTriggered()));
	connect(ui.actionSave, SIGNAL(triggered()), this, SLOT(saveTriggered()));
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(exitTriggered()));
	connect(ui.actionOpenCL, SIGNAL(triggered(bool)), this, SLOT(openCLTriggered(bool)));

	connect(ui.cbInvert, SIGNAL(stateChanged(int)), this, SLOT(invertChanged(int)));

	// Operacje
	connect(ui.rbNone, SIGNAL(toggled(bool)), this, SLOT(noneOperationToggled(bool)));
	connect(ui.rbErode, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbDilate, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbOpen, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbClose, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbGradient, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbTopHat, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbBlackHat, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbOutline, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbSkeleton, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbSkeletonZhang, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));

	// Element strukturalny
	connect(ui.rbRect, SIGNAL(toggled(bool)), this, SLOT(structuringElementToggled(bool)));
	connect(ui.rbEllipse, SIGNAL(toggled(bool)), this, SLOT(structuringElementToggled(bool)));
	connect(ui.rbCross, SIGNAL(toggled(bool)), this, SLOT(structuringElementToggled(bool)));
	connect(ui.rbDiamond, SIGNAL(toggled(bool)), this, SLOT(structuringElementToggled(bool)));

	connect(ui.pbShowSE, SIGNAL(pressed()), this, SLOT(structuringElementPreview()));

	// Rozmiar elementu strukturalnego
	connect(ui.cbSquare, SIGNAL(stateChanged(int)), this, SLOT(ratioChanged(int)));
	connect(ui.hsXElementSize, SIGNAL(valueChanged(int)), this, SLOT(elementSizeXChanged(int)));
	connect(ui.hsYElementSize, SIGNAL(valueChanged(int)), this, SLOT(elementSizeYChanged(int)));
	connect(ui.dialRotation, SIGNAL(valueChanged(int)), this, SLOT(rotationChanged(int)));
	connect(ui.pbResetRotation, SIGNAL(pressed()), this, SLOT(rotationResetPressed()));

	connect(ui.cbAutoTrigger, SIGNAL(stateChanged(int)), this, SLOT(autoRunChanged(int)));
	connect(ui.pbRun, SIGNAL(pressed()), this, SLOT(runPressed()));

	QSettings settings("./settings.cfg", QSettings::IniFormat);
	maxImageWidth = settings.value("gui/maximagewidth", 512).toInt();
	maxImageHeight = settings.value("gui/maximageheight", 512).toInt();
	
	int method = 0;
	printf("There are 2 methods implemented:\n"
		"\t1) Images\n"
		"\t2) Buffers\n");
	while (method != 1 && method != 2)
	{
		printf("Choose method: ");
		int r = scanf("%d", &method);
		// Jesli nie odczytano jednej liczby (np. wprowadzono znak A)
		// trzeba opronznic stdin, inaczej wpadniemy w nieskonczona petle
		if(r != 1)
		{
			char buf[128];
			fgets(buf, 128, stdin);
		}
	}

	if (method == 1) ocl = new MorphOpenCLImage();
	else ocl = new MorphOpenCLBuffer();

	ocl->errorCallback = [this](const QString& message, cl_int err)
	{
		Q_UNUSED(err);
		QMessageBox::critical(this, "OpenCL error", 
			QString("%1\nError: %2").arg(message).arg(ocl->openCLErrorCodeStr(err)),
			QMessageBox::Ok);
		exit(1);
	};

	oclSupported = ocl->initOpenCL();
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
	ui.rbEllipse->toggle();
	ui.cbSquare->setChecked(true);

	ui.lbXElementSize->setText(QString::fromLatin1("Horizontal: 3"));
	ui.lbYElementSize->setText(QString::fromLatin1("Vertical: 3"));
	ui.lbRotation->setText(QString::fromLatin1("0"));

	statusBarLabel = new QLabel();
	ui.statusBar->addPermanentWidget(statusBarLabel);
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

		if(ui.cbAutoTrigger->isChecked() || ui.rbNone->isChecked())
			refresh();
		else
			ui.rbNone->setChecked(true);

		resize(1, 1);
	}
}
// -------------------------------------------------------------------------
void MainWindow::saveTriggered()
{
	QString filename = QFileDialog::getSaveFileName(this, QString(), ".",
		QString::fromLatin1("Image file (*.png)"));

	if(!filename.isEmpty())
	{
		//ui.lbImage->pixmap()->toImage().save(filename);
		cv::Mat dstc;
		cvtColor(dst, dstc, CV_GRAY2BGR);

		QImage qdst(
			reinterpret_cast<const quint8*>(dstc.data),
			dstc.cols, dstc.rows, dstc.step, 
			QImage::Format_RGB888);
		qdst.save(filename);
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

	if(ui.cbAutoTrigger->isChecked())
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
void MainWindow::noneOperationToggled(bool checked)
{
	if(checked)
	{
		// aktywuj wybor elementu strukturalnego
		ui.gbElement->setEnabled(true);
		ui.gbElementSize->setEnabled(true);

		refresh();
		ui.pbRun->setEnabled(false);
		ui.actionSave->setEnabled(false);
	}
	else
	{
		ui.pbRun->setEnabled(true);
	}
}
// -------------------------------------------------------------------------
void MainWindow::operationToggled(bool checked)
{
	// Warunek musi byc spelniony bo sa zglaszane 2 zdarzenia
	// 1 - jeden z radiobuttonow zmienil stan z aktywnego na nieaktywny
	// 2 - zaznaczony radiobutton zmienil stan z nieaktywnego na aktywny
	if(checked)
	{
		// Operacje hit-miss
		if (ui.rbOutline->isChecked() ||
			ui.rbSkeleton->isChecked() ||
			ui.rbSkeletonZhang->isChecked())
		{
			// deaktywuj wybor elementu strukturalnego
			ui.gbElement->setEnabled(false);
			ui.gbElementSize->setEnabled(false);
		}
		else
		{
			// aktywuj wybor elementu strukturalnego
			ui.gbElement->setEnabled(true);
			ui.gbElementSize->setEnabled(true);
		}

		if(ui.cbAutoTrigger->isChecked())
			refresh();
	}
}
// -------------------------------------------------------------------------
void MainWindow::structuringElementToggled(bool checked)
{
	if(checked)
	{
		if(!ui.rbNone->isChecked() && ui.cbAutoTrigger->isChecked())
			refresh();
	}
}
// -------------------------------------------------------------------------
void MainWindow::structuringElementPreview()
{
	QDialog* d = new QDialog(this);

	Ui::SEPreview uid;
	uid.setupUi(d);

	cv::Mat se = standardStructuringElement();
	// Konwertuje 0,1 na 0,255
	cv::Mat lut(1, 256, CV_8U);
	uchar* p = lut.ptr<uchar>();
	p[0] = 0;
	for(int i = 1; i < lut.cols; ++i) 
		p[i] = 255;
	cv::LUT(se, lut, se);

	int xsize = 256;
	int ysize = 256;
	if(se.cols != se.rows)
	{
		auto round = [](double v) { return static_cast<int>(v + 0.5); };

		if(se.cols > se.rows) ysize = se.rows * round(256.0/se.cols);
		else xsize = se.cols * round(256.0/se.rows);
	}
	cv::resize(se, se, cv::Size(xsize, ysize), 0.0, 0.0, cv::INTER_NEAREST);

	QImage img(reinterpret_cast<const quint8*>(se.data),
		se.cols, se.rows, se.step, 
		QImage::Format_Indexed8);

	uid.lbSEPreview->setPixmap(QPixmap::fromImage(img));

	d->setModal(true);
	d->exec();
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
	kradiusx = ui.hsXElementSize->value();

	if (ui.cbSquare->checkState() == Qt::Checked)
	{
		if (ui.hsYElementSize->value() != ui.hsXElementSize->value())
		{
			disableRefreshing = true;
			ui.hsYElementSize->setValue(ui.hsXElementSize->value());
		}
	}	

	if (!ui.rbNone->isChecked() && 
		ui.cbAutoTrigger->isChecked() &&
		!disableRefreshing)
	{
		refresh();
	}

	disableRefreshing = false;
}
// -------------------------------------------------------------------------
void MainWindow::elementSizeYChanged(int value)
{
	Q_UNUSED(value);
	ui.lbYElementSize->setText(QString::fromLatin1("Vertical: ") + 
		QString::number(2 * ui.hsYElementSize->value() + 1));
	kradiusy = ui.hsYElementSize->value();

	if (ui.cbSquare->checkState() == Qt::Checked)
	{
		if (ui.hsXElementSize->value() != ui.hsYElementSize->value())
		{
			disableRefreshing = true;
			ui.hsXElementSize->setValue(ui.hsYElementSize->value());
		}
	}	

	if (!ui.rbNone->isChecked() && 
		ui.cbAutoTrigger->isChecked() &&
		!disableRefreshing)
	{
		refresh();
	}

	disableRefreshing = false;
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

	if(!ui.rbNone->isChecked() && ui.cbAutoTrigger->isChecked())
		refresh();
}
// -------------------------------------------------------------------------
void MainWindow::rotationResetPressed()
{
	ui.dialRotation->setValue(180);
}
// -------------------------------------------------------------------------
void MainWindow::runPressed()
{
	refresh();
}
// -------------------------------------------------------------------------
void MainWindow::autoRunChanged(int state)
{
	if(state == Qt::Checked)
		refresh();
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

	cv::Mat img = image;
	if(img.rows > maxImageHeight ||img.cols > maxImageWidth)
	{
		double fx;
		if(img.rows > img.cols)
			fx = static_cast<double>(maxImageHeight) / img.rows;
		else
			fx = static_cast<double>(maxImageWidth) / img.cols;
		cv::resize(img, img, cv::Size(0,0), fx, fx, cv::INTER_LINEAR);
	}

	ui.lbImage->setPixmap(QPixmap::fromImage(toQImage(img)));
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
}
// -------------------------------------------------------------------------
void MainWindow::refresh()
{
	if(ui.rbNone->isChecked())
	{
		showCvImage(src);
		return;
	}

	if(ui.actionOpenCL->isChecked())
		morphologyOpenCL();
	else
		morphologyOpenCV();

	ui.actionSave->setEnabled(true);
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

	// Operacje hit-miss
	if (opType == OT_Outline ||
		opType == OT_Skeleton ||
		opType == OT_Skeleton_ZhangSuen)
	{
		switch (opType)
		{
		case OT_Outline:
			{
				morphologyOutline(src, dst);
				break;
			}
		case OT_Skeleton:
			{
				iters = morphologySkeleton(src, dst);
				break;
			}
		case OT_Skeleton_ZhangSuen:
			{
				iters = morphologySkeletonZhangSuen(src, dst);
				break;
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
		cv::morphologyEx(src, dst, op_type, element);
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
	strm << "Time elapsed: " << elapsed << " ms, iterations: " << iters;
	printf("Time elapsed: %lf ms, iterations: %d\n", elapsed, iters);
	statusBarLabel->setText(txt);
}
// -------------------------------------------------------------------------
void MainWindow::morphologyOpenCL()
{
	cv::Mat element = standardStructuringElement();
	EOperationType opType = operationType();

	int iters;
	int csize = ocl->setStructuringElement(element);
	double delapsed = ocl->morphology(opType, dst, iters);
	
	// Wyswietl statystyki
	QString txt; 
	QTextStream strm(&txt);
	strm << "Time elapsed : " << delapsed << " ms, iterations: " << iters;
	statusBarLabel->setText(txt);

	// Pokaz obraz wynikowy
	showCvImage(dst);
}
// -------------------------------------------------------------------------
cv::Mat MainWindow::standardStructuringElement()
{
	EStructuringElementType type;

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
	else if(ui.rbOutline->isChecked()) { return OT_Outline; }
	else if(ui.rbSkeleton->isChecked()) { return OT_Skeleton; }
	else if(ui.rbSkeletonZhang->isChecked()) { return OT_Skeleton_ZhangSuen; }
	else { return OT_Erode; }
}
