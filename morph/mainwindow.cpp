#include "mainwindow.h"

#include <QElapsedTimer>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>
#include <QSettings>

#include <omp.h>

#if !defined(_WIN32)
#include <sys/time.h>
#endif

#include "ui_sepreview.h"
#include "ui_settings.h"

#include "cvutils.h"
#include "morphoclbuffer.h"
#include "morphoclimage.h"
#include "morphoperators.h"

MainWindow::MainWindow(QString filename, QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags),
	disableRefreshing(false),
	krotation(0)
{
	ui.setupUi(this);

	// Menu
	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(openTriggered()));
	connect(ui.actionSave, SIGNAL(triggered()), this, SLOT(saveTriggered()));
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(exitTriggered()));
	connect(ui.actionOpenCL, SIGNAL(triggered(bool)), this, SLOT(openCLTriggered(bool)));
	connect(ui.actionPickMethod, SIGNAL(triggered()), this, SLOT(pickMethodTriggered()));
	connect(ui.actionSettings, SIGNAL(triggered()), this, SLOT(settingsTriggered()));
	connect(ui.actionCameraInput, SIGNAL(triggered(bool)), this, SLOT(cameraInputTriggered(bool)));
	connect(ui.actionOpenSE, SIGNAL(triggered()), this, SLOT(openSETriggered()));
	connect(ui.actionSaveSE, SIGNAL(triggered()), this, SLOT(saveSETriggered()));

	connect(ui.cbInvert, SIGNAL(stateChanged(int)), this, SLOT(invertChanged(int)));
	connect(ui.cmbBayer, SIGNAL(currentIndexChanged(int)), this, SLOT(bayerIndexChanged(int)));

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

	// Wymuszenie initializeGL tak aby miec juz utworzony kontekst OGL
	// OpenGL najpierw, pozniej OpenCL
	ui.glWidget->updateGL();
	ui.glWidget->makeCurrent();

	QSettings settings("./settings.cfg", QSettings::IniFormat);
	maxImageWidth = settings.value("gui/maximagewidth", 512).toInt();
	maxImageHeight = settings.value("gui/maximageheight", 512).toInt();
	
	method = 0;
	printf("There are 2 methods implemented:\n"
		"\t1) 2D buffer (image object)\n"
		"\t2) 1D buffer (buffer object)\n");
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
	initOpenCL(method);
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
		if(ui.actionCameraInput->isChecked())
		{
			ui.actionCameraInput->setChecked(false);
			cameraInputTriggered(false);
		}

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
void MainWindow::cameraInputTriggered(bool state)
{
	if(state)
	{
		camera.open(0);
		if(!camera.isOpened())
		{
			QMessageBox::critical(nullptr, "Error", 
				"Cannot establish connection to default camera device.", QMessageBox::Ok);
			ui.actionCameraInput->setChecked(false);
			return;
		}

		printf("Connection established\n");
		printf("Frame size: %.0fx%.0f\n", 
			camera.get(CV_CAP_PROP_FRAME_WIDTH),
			camera.get(CV_CAP_PROP_FRAME_HEIGHT));

		// CV_CAP_PROP_FORMAT zwraca tylko format danych (np. CV_8U), bez liczby kanalow
		cv::Mat dummy;
		camera.read(dummy);
		int type = dummy.type();

		const char* s_type;
		switch(type)
		{
#define ENUM_TO_STR(X) case X: s_type = #X; break;
		ENUM_TO_STR(CV_8UC1);
		ENUM_TO_STR(CV_8UC2);
		ENUM_TO_STR(CV_8UC3);
		ENUM_TO_STR(CV_8UC4);
		ENUM_TO_STR(CV_8SC1);
		ENUM_TO_STR(CV_8SC2);
		ENUM_TO_STR(CV_8SC3);
		ENUM_TO_STR(CV_8SC4);
		ENUM_TO_STR(CV_16UC1);
		ENUM_TO_STR(CV_16UC2);
		ENUM_TO_STR(CV_16UC3);
		ENUM_TO_STR(CV_16UC4);
		ENUM_TO_STR(CV_16SC1);
		ENUM_TO_STR(CV_16SC2);
		ENUM_TO_STR(CV_16SC3);
		ENUM_TO_STR(CV_16SC4);
		default:
			s_type = "(unknown format)"; break;
#undef ENUM_TO_STR
		}

		printf("Camera input format: %s\n", s_type);

		timerId = startTimer(100);
	}
	else
	{
		killTimer(timerId);
		camera.release();
	}
}
// -------------------------------------------------------------------------
void MainWindow::openSETriggered()
{
	QString filename = QFileDialog::getOpenFileName(this, QString(), ".",
		QString::fromLatin1("Structuring element file (*.se)"));

	if(!filename.isEmpty())
	{
		// Dane do deserializacji
		Morphology::EStructuringElementType etype;
		int xradius, yradius, rotation, type;
		unsigned magic;

		// Otworz wskazany plik i go zdeserializuj
		QFile file(filename);
		file.open(QIODevice::ReadOnly);
		QDataStream strm(&file);
		strm >> magic;
		if(magic != 0x1337U)
		{
			QMessageBox::critical(nullptr, "Error",
				"Unknown format file or file is corrupted.",
				QMessageBox::Ok);
			return;
		}

		strm >> type >> xradius >> yradius >> rotation;
		etype = static_cast<Morphology::EStructuringElementType>(type);
		file.close();

		// Jesli mamy wlaczonego auto-refresha, deaktywujemy go na chwile
		bool autorefresh = ui.cbAutoTrigger->isChecked();
		ui.cbAutoTrigger->setChecked(false);

		// Ustaw ksztalt elementu strukturalnego
		switch(etype)
		{
		case Morphology::SET_Rect:
			ui.rbRect->setChecked(true); break;
		case Morphology::SET_Ellipse:
			ui.rbEllipse->setChecked(true); break;
		case Morphology::SET_Cross:
			ui.rbCross->setChecked(true); break;
		case Morphology::SET_Diamond:
		default:
			ui.rbDiamond->setChecked(true); break;
		}

		// Ustaw jego rozmiar oraz rotacje
		if(xradius != yradius)
			ui.cbSquare->setChecked(false);
		else
			ui.cbSquare->setChecked(true);

		ui.hsXElementSize->setValue(xradius);
		ui.hsYElementSize->setValue(yradius);
		ui.dialRotation->setValue(rotation);

		// Przywroc auto-refresha to stanu poprzedniego
		// Jesli byl ustalony to zostanie wywolany refresh()
		ui.cbAutoTrigger->setChecked(autorefresh);
	}
}
// -------------------------------------------------------------------------
void MainWindow::saveSETriggered()
{
	QString filename = QFileDialog::getSaveFileName(this, QString(), ".",
		QString::fromLatin1("Structuring element file (*.se)"));

	if(!filename.isEmpty())
	{
		Morphology::EStructuringElementType type;
		if(ui.rbRect->isChecked()) type = Morphology::SET_Rect;
		else if(ui.rbEllipse->isChecked()) type = Morphology::SET_Ellipse;
		else if(ui.rbCross->isChecked()) type = Morphology::SET_Cross;
		else type = Morphology::SET_Diamond;

		int xradius = ui.hsXElementSize->value();
		int yradius = ui.hsYElementSize->value();
		int rotation = ui.dialRotation->value();

		QFile file(filename);
		file.open(QIODevice::WriteOnly);
		QDataStream strm(&file);
		strm << 0x1337U << type << xradius << yradius << rotation;
		file.close();
	}
}
// -------------------------------------------------------------------------
void MainWindow::pickMethodTriggered()
{
	QMessageBox msgBox;
	msgBox.setText("Choose different method:");
	QPushButton* buffer1D = msgBox.addButton("Buffer1D", QMessageBox::AcceptRole);
	QPushButton* buffer2D = msgBox.addButton("Buffer2D", QMessageBox::AcceptRole);
	QPushButton* cancel = msgBox.addButton(QMessageBox::Cancel);
	msgBox.setDefaultButton(cancel);

	msgBox.exec();

	if(msgBox.clickedButton() == buffer1D)
		method = 2;
	else if(msgBox.clickedButton() == buffer2D)
		method = 1;
	else
		return;

 	delete ocl;
 	initOpenCL(method);
 	setOpenCLSourceImage();
}
// -------------------------------------------------------------------------
void MainWindow::settingsTriggered()
{
	QDialog* d = new QDialog(this);
	d->setModal(true);

	Ui::SettingDialog uid;
	uid.setupUi(d);

	// Ustaw wartosci kontrolek zgodnie z ustawieniami w settings.cfg
	QSettings s("./settings.cfg", QSettings::IniFormat);
	auto setComboBoxIndex = [&s](QComboBox* cb, const QString& path)
	{
		int i = cb->findText(s.value(path).toString());
		//if(i == -1) i = 0;
		cb->setCurrentIndex(i);
	};

	// Sekcja Preview
	uid.maxImageWidthLineEdit->setText(s.value("gui/maximagewidth").toString());
	uid.maxImageHeightLineEdit->setText(s.value("gui/maximageheight").toString());
	uid.defaultImageLineEdit->setText(s.value("gui/defaultimage").toString());

	// Sekcja OpenCL
	uid.useAtomicCountersCheckBox->setChecked(s.value("opencl/atomiccounters").toBool());
	uid.openGInteropCheckBox->setChecked(s.value("opencl/glinterop").toBool());
	uid.datatypeComboBox->setCurrentIndex(s.value("opencl/datatype").toInt());
	setComboBoxIndex(uid.workgroupSizeXComboBox, "opencl/workgroupsizex");
	setComboBoxIndex(uid.workgroupSizeYComboBox, "opencl/workgroupsizey");

	// Sekcja Buffer 2D kernels
	setComboBoxIndex(uid.erodeKernelComboBox, "kernel-buffer2D/erode");
	setComboBoxIndex(uid.dilateKernelComboBox, "kernel-buffer2D/dilate");
	setComboBoxIndex(uid.gradientKernelComboBox, "kernel-buffer2D/gradient");

	// Sekcja Buffer 1D kernels
	setComboBoxIndex(uid.erodeKernelComboBox_2, "kernel-buffer1D/erode");
	setComboBoxIndex(uid.dilateKernelComboBox_2, "kernel-buffer1D/dilate");
	setComboBoxIndex(uid.gradientKernelComboBox_2, "kernel-buffer1D/gradient");
	setComboBoxIndex(uid.subtractKernelComboBox, "kernel-buffer1D/subtract");
	setComboBoxIndex(uid.hitmissMemTypeComboBox, "kernel-buffer1D/hitmiss");

	// Ustaw dodatkowo walidator
	QIntValidator validator(0, 2048);
	uid.maxImageWidthLineEdit->setValidator(&validator);
	uid.maxImageHeightLineEdit->setValidator(&validator);

	int ret = d->exec();
	if(ret == QDialog::Accepted)
	{
		s.setValue("gui/maximagewidth", uid.maxImageWidthLineEdit->text());
		s.setValue("gui/maximageheight", uid.maxImageHeightLineEdit->text());
		s.setValue("gui/defaultimage", uid.defaultImageLineEdit->text());

		s.setValue("opencl/atomiccounters", 
			QVariant(uid.useAtomicCountersCheckBox->isChecked()));
		s.setValue("opencl/glinterop", 
			QVariant(uid.openGInteropCheckBox->isChecked()));
		s.setValue("opencl/datatype", uid.datatypeComboBox->currentText());
		s.setValue("opencl/workgroupsizex", uid.workgroupSizeXComboBox->currentText());
		s.setValue("opencl/workgroupsizey", uid.workgroupSizeYComboBox->currentText());

		s.setValue("kernel-buffer2D/erode", uid.erodeKernelComboBox->currentText());
		s.setValue("kernel-buffer2D/dilate", uid.dilateKernelComboBox->currentText());
		s.setValue("kernel-buffer2D/gradient", uid.gradientKernelComboBox->currentText());

		s.setValue("kernel-buffer1D/erode", uid.erodeKernelComboBox_2->currentText());
		s.setValue("kernel-buffer1D/dilate", uid.dilateKernelComboBox_2->currentText());
		s.setValue("kernel-buffer1D/gradient", uid.gradientKernelComboBox_2->currentText());
		s.setValue("kernel-buffer1D/subtract", uid.subtractKernelComboBox->currentText());
		s.setValue("kernel-buffer1D/hitmiss", uid.hitmissMemTypeComboBox->currentText());

		QMessageBox::information(this, "Settings", 
			"You need to restart the application to apply changes.", QMessageBox::Ok);
	}
}
// -------------------------------------------------------------------------
void MainWindow::invertChanged(int state)
{
	Q_UNUSED(state);
	
	CvUtil::negateImage(src);
	setOpenCLSourceImage();
	refresh();
}
// -------------------------------------------------------------------------
void MainWindow::bayerIndexChanged(int i)
{
	if(oclSupported)
		ocl->setBayerFilter(static_cast<Morphology::EBayerCode>(i));
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
// -------------------------------------------------------------------------
void MainWindow::timerEvent(QTimerEvent* event)
{
	Q_UNUSED(event);

	camera >> src;

	// TODO: hardcoded
	if(src.channels() != 1)
		cvtColor(src, src, CV_BGR2GRAY);

	if(ui.cbInvert->isChecked())
		CvUtil::negateImage(src);

	setOpenCLSourceImage();
	refresh();
}

// Koniec zdarzen
// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH

void MainWindow::initOpenCL(int method)
{
	if (method == 1) ocl = new MorphOpenCLImage();
	else ocl = new MorphOpenCLBuffer();

	ocl->setErrorCallback([this](const QString& message, cl_int err)
	{
		Q_UNUSED(err);
		QMessageBox::critical(this, "OpenCL error",
			QString("%1\nError: %2").arg(message).arg(ocl->openCLErrorCodeStr(err)),
			QMessageBox::Ok);
		oclSupported = false;
	});

	oclSupported = true;
	oclSupported &= ocl->initOpenCL();

	ocl->setErrorCallback([this](const QString& message, cl_int err)
	{
		Q_UNUSED(err);
		QMessageBox::critical(this, "OpenCL error",
			QString("%1\nError: %2").arg(message).arg(ocl->openCLErrorCodeStr(err)),
			QMessageBox::Ok);
	});

	if(oclSupported)
	{
		ui.actionOpenCL->setEnabled(true);
		ui.actionOpenCL->setChecked(true);
	}
	else
	{
		QMessageBox::critical(nullptr,
			"Critical error",
			"No OpenCL Platform available or something terrible happened "
			"during OpenCL initialization therefore OpenCL processing will be disabled.",
			QMessageBox::Ok);

		ui.actionOpenCL->setEnabled(false);
		ui.actionOpenCL->setChecked(false);
	}	
}
// -------------------------------------------------------------------------
void MainWindow::showCvImage(const cv::Mat& image)
{
	QSize surfaceSize(image.cols, image.rows);

	if(image.rows > maxImageHeight ||image.cols > maxImageWidth)
	{
		double fx;
		if(image.rows > image.cols)
			fx = static_cast<double>(maxImageHeight) / image.rows;
		else
			fx = static_cast<double>(maxImageWidth) / image.cols;

		surfaceSize.setWidth(image.cols * fx);
		surfaceSize.setHeight(image.rows * fx);
	}

	ui.glWidget->setMinimumSize(surfaceSize);
	ui.glWidget->setMaximumSize(surfaceSize);
	ui.glWidget->setSurface(image);

	adjustSize();
}
// -------------------------------------------------------------------------
void MainWindow::showGlImage(int w, int h)
{
	QSize surfaceSize(w, h);

	if(h > maxImageHeight || w > maxImageWidth)
	{
		double fx;
		if(h > w)
			fx = static_cast<double>(maxImageHeight) / h;
		else
			fx = static_cast<double>(maxImageWidth) / w;

		surfaceSize.setWidth(w * fx);
		surfaceSize.setHeight(h * fx);
	}

	ui.glWidget->setMinimumSize(surfaceSize);
	ui.glWidget->setMaximumSize(surfaceSize);
	ui.glWidget->updateGL();

	adjustSize();
}
// -------------------------------------------------------------------------
void MainWindow::openFile(const QString& filename)
{
	src = cv::imread(filename.toStdString());
	int depth = src.depth();
	int channels = src.channels();

	//printf("depth:%d channels:%d\n", depth, channels);

	Q_ASSERT(depth == CV_8U);

	if(channels == 3)
		cvtColor(src, src, CV_BGR2GRAY);
	else if(channels == 4)
		cvtColor(src, src, CV_BGRA2GRAY);

	setOpenCLSourceImage();
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
	Morphology::EOperationType opType = operationType();
	cv::Mat src_ = src;

	if(ui.cmbBayer->currentIndex() != 0)
	{
		// Jest bug dla CV_BayerXX2GRAY i trzeba wykonac sciezke okrezna
		switch(ui.cmbBayer->currentIndex())
		{
		case 1: cv::cvtColor(src, src_, CV_BayerRG2BGR); break;
		case 2: cv::cvtColor(src, src_, CV_BayerGR2BGR); break;
		case 3: cv::cvtColor(src, src_, CV_BayerBG2BGR); break;
		case 4: cv::cvtColor(src, src_, CV_BayerGB2BGR); break;
		default: break;
		}
		cvtColor(src_, src_, CV_BGR2GRAY);
	}

	// Operacje hit-miss
	if (opType == Morphology::OT_Outline ||
		opType == Morphology::OT_Skeleton ||
		opType == Morphology::OT_Skeleton_ZhangSuen)
	{
		switch (opType)
		{
		case Morphology::OT_Outline:
			{
				Morphology::outline(src_, dst);
				break;
			}
		case Morphology::OT_Skeleton:
			{
				iters = Morphology::skeleton(src_, dst);
				break;
			}
		case Morphology::OT_Skeleton_ZhangSuen:
			{
				iters = Morphology::skeletonZhangSuen(src_, dst);
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
		case Morphology::OT_Erode: op_type = cv::MORPH_ERODE; break;
		case Morphology::OT_Dilate: op_type = cv::MORPH_DILATE; break;
		case Morphology::OT_Open: op_type = cv::MORPH_OPEN; break;
		case Morphology::OT_Close: op_type = cv::MORPH_CLOSE; break;
		case Morphology::OT_Gradient: op_type = cv::MORPH_GRADIENT; break;
		case Morphology::OT_TopHat: op_type = cv::MORPH_TOPHAT; break;
		case Morphology::OT_BlackHat: op_type = cv::MORPH_BLACKHAT; break;
		default: op_type = cv::MORPH_ERODE; break;
		}

		cv::Mat element = standardStructuringElement();
		cv::morphologyEx(src_, dst, op_type, element);
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
	Morphology::EOperationType opType = operationType();
	int iters;
	
	ocl->error = false;
	int csize = ocl->setStructuringElement(element);

	if(!ocl->error) 
	{
		ocl->recompile(opType, csize);
	}

	if(!ocl->error) 
	{
		double delapsed = ocl->morphology(opType, dst, iters);

		// Wyswietl statystyki
		QString txt; 
		QTextStream strm(&txt);
		strm << "Time elapsed : " << delapsed << " ms, iterations: " << iters;
		statusBarLabel->setText(txt);

		// Pokaz obraz wynikowy
		if(!ocl->usingShared())
			showCvImage(dst);
		else
			showGlImage(src.cols, src.rows);
	}	
}
// -------------------------------------------------------------------------
cv::Mat MainWindow::standardStructuringElement()
{
	using namespace Morphology;
	EStructuringElementType type;

	if(ui.rbRect->isChecked()) type = SET_Rect;
	else if(ui.rbEllipse->isChecked()) type = SET_Ellipse;
	else if(ui.rbCross->isChecked()) type = SET_Cross;
	else type = SET_Diamond;

	return Morphology::standardStructuringElement(
		ui.hsXElementSize->value(),
		ui.hsYElementSize->value(),
		type, krotation);
}
// -------------------------------------------------------------------------
Morphology::EOperationType MainWindow::operationType()
{
	using namespace Morphology;

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
// -------------------------------------------------------------------------
void MainWindow::setOpenCLSourceImage()
{
	if(oclSupported)
	{
		if(ocl->usingShared())
		{
			GLuint glresource = ui.glWidget->createEmptySurface
				(src.cols, src.rows);
			ocl->setSourceImage(&src, glresource);
		}
		else
		{
			ocl->setSourceImage(&src);
		}
	}
}
