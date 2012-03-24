#include "controller.h"

#include <QFileDialog>
#include <QMessageBox>

#include "cvutils.h"
#include "elapsedtimer.h"
#include "settings.h"
#include "sepreview.h"

#define USE_GLWIDGET 0

Controller* Controller::msSingleton = nullptr;

Controller::Controller()
	: mw(nullptr)
	, ocl(nullptr)
	, oclSupported(false)
	, useOpenCL(false)
	, autoTrigger(false)
	, resizeCustomSe(true)
{
}

Controller::~Controller()
{
	delete mw;
	delete ocl;
}

void Controller::start()
{
	conf.loadConfiguration("./settings.cfg");
	QString defImage(conf.defaultImage);

	// Wczytaj domyslny obraz (jesli wyspecyfikowano)
	if(defImage.isEmpty())
	{
		defImage = QFileDialog::getOpenFileName(mw, QString(), ".",
			QLatin1String("Image files (*.png *.jpg *.bmp)"));

		if(defImage.isEmpty())
			exit(-1);
	}

	mw = new MainWindow;

#if USE_GLWIDGET == 1
	previewWidget = new GLWidget(mw->centralWidget());
	previewWidget->setMinimumSize(QSize(10, 10));
	mw->setPreviewWidget(previewWidget);

	previewWidget->updateGL();
	previewWidget->makeCurrent();
#else
	// Widget wyswietlajacy dany obraz
	previewLabel = new QLabel(mw->centralWidget());
	previewLabel->setText(QString());
	previewLabel->setMinimumSize(QSize(10, 10));

	mw->setPreviewWidget(previewLabel);
#endif

	//initializeOpenCL(OM_Buffer1D);
	mw->setOpenCLCheckableAndChecked(false);

	connect(mw, SIGNAL(recomputeNeeded()), this, SLOT(onRecompute()));
	connect(mw, SIGNAL(sourceImageShowed()), this, SLOT(onShowSourceImage()));
	connect(mw, SIGNAL(structuringElementChanged()), this, SLOT(onStructuringElementChanged()));

	openFile(defImage);
	onShowSourceImage();

	mw->show();
}

void Controller::onFromCameraTriggered(bool state)
{

}

void Controller::onOpenFileTriggered()
{
	QString filename = QFileDialog::getOpenFileName(mw, QString(), ".",
		QLatin1String("Image files (*.png *.jpg *.bmp)"));

	if(filename.isEmpty())
		return;

	openFile(filename);

	if(autoTrigger && !mw->isNoneOperationChecked())
	{
		onRecompute();
	}
	else
	{
		mw->setMorphologyOperation(Morphology::OT_None);
		onShowSourceImage();
	}	

	mw->resize(1, 1);
}

void Controller::onSaveFileTriggered()
{
	QString filename = QFileDialog::getSaveFileName(mw, QString(), ".",
		QLatin1String("Image file (*.png)"));

	if(filename.isEmpty())
		return;

	cv::Mat dstc; cvtColor(dst, dstc, CV_GRAY2BGR);
	cv::imwrite(filename.toStdString(), dstc);
}

void Controller::onOpenStructuringElementTriggered()
{
	QString filename = QFileDialog::getOpenFileName(mw, QString(), ".",
		QLatin1String("Structuring element file (*.se)"));
	
	if(filename.isEmpty())
		return;

	// Dane do deserializacji
	Morphology::EStructuringElementType etype;
	int rotation, type;
	QSize ksize;
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

	strm >> type;
	etype = static_cast<Morphology::EStructuringElementType>(type);

	if(etype != Morphology::SET_Custom)
	{
		strm >> ksize >> rotation;
		
		mw->setStructuringElementType(etype);
		mw->setStructuringElementSize(ksize);
		mw->setStructuringElementRotation(rotation);
	}
	else
	{
		int w, h;
		strm >> w >> h;
		
		customSe = cv::Mat(cv::Size(w, h), CV_8UC1);

		char* ptr = customSe.ptr<char>(0);
		for(int i = 0; i < customSe.rows; ++i)
		{
			strm.readRawData(ptr, customSe.cols);
			ptr += customSe.step1();
		}

		ksize = QSize(w-1, h-1) / 2;

		resizeCustomSe = false;
		mw->setStructuringElementType(etype);
		mw->setStructuringElementSize(ksize);
		resizeCustomSe = true;
	}	

	file.close();
}

void Controller::onSaveStructuringElementTriggered()
{
	QString filename = QFileDialog::getSaveFileName(mw, QString(), ".",
		QLatin1String("Structuring element file (*.se)"));

	if(filename.isEmpty())
		return;

	QFile file(filename);
	file.open(QIODevice::WriteOnly);
	QDataStream strm(&file);

	auto type = mw->structuringElementType();
	strm << 0x1337U << type;

	if(type != Morphology::SET_Custom)
	{
		QSize ksize = mw->structuringElementSize();
		int rotation = mw->structuringElementRotation();

		strm << ksize << rotation;
	}
	else
	{
		strm << customSe.cols << customSe.rows;

		// Musimy zapisywac rzad po rzedzie - zdaje sie, 
		// ze OpenCV przechowuje male obrazy z wyrownaniem do 32 bajtow
		const char* ptr = customSe.ptr<char>(0);
		for(int i = 0; i < customSe.rows; ++i)
		{
			strm.writeRawData(ptr, customSe.cols);
			ptr += customSe.step1();
		}
	}

	file.close();
}

void Controller::onInvertChanged(int state)
{
	Q_UNUSED(state);

	CvUtil::negateImage(src);
	setOpenCLSourceImage();

	if(mw->isNoneOperationChecked())
		onShowSourceImage();
	else if(autoTrigger)
		onRecompute();
}

void Controller::onOpenCLTriggered(bool state)
{
	useOpenCL = state;

	if(autoTrigger)
		onRecompute();
}

void Controller::onPickMethodTriggerd()
{
	QMessageBox msgBox;
	QPushButton* buffer1D = msgBox.addButton("Buffer1D", QMessageBox::AcceptRole);
	QPushButton* buffer2D = msgBox.addButton("Buffer2D", QMessageBox::AcceptRole);
	QPushButton* cancel = msgBox.addButton(QMessageBox::Cancel);

	msgBox.setWindowTitle("Morph OpenCL");
	msgBox.setText("Choose different method:");
	msgBox.setDefaultButton(cancel);
	msgBox.exec();

	EOpenCLMethod method;

	// Co wybrano
	if(msgBox.clickedButton() == buffer1D)
		method = OM_Buffer1D;
	else if(msgBox.clickedButton() == buffer2D)
		method = OM_Buffer2D;
	else
		return;

	// Reinicjalizuj modul OpenCLa
	delete ocl;
	initializeOpenCL(method);
	setOpenCLSourceImage();
}

void Controller::onSettingsTriggered()
{
	Settings* s = new Settings(mw);
	s->setConfigurationModel(conf);

	int ret = s->exec();
	if(ret != QDialog::Accepted)
		return;

	Configuration conf = s->configurationModel();
	conf.saveConfiguration("./settings.cfg");		

	QMessageBox::information(mw, "Settings", 
		"You need to restart the application to apply changes.", QMessageBox::Ok);
}

void Controller::onAutoTriggerChanged(int state)
{
	if(state == Qt::Checked)
	{
		autoTrigger = true;
		onRecompute();
	}
	else
	{
		autoTrigger = false;
	}
}

void Controller::onBayerIndexChanged(int bcode)
{
	if(oclSupported)
		ocl->setBayerFilter(static_cast<Morphology::EBayerCode>(bcode));

	if(autoTrigger)
		onRecompute();
}

void Controller::onStructuringElementChanged()
{
	if(mw->structuringElementType() != Morphology::SET_Custom)
	{
		mw->setEnabledStructuringElementRotation(true);
		cv::Mat se(standardStructuringElement());
		emit structuringElementChanged(se);
	}
	else
	{
		mw->setEnabledStructuringElementRotation(false);

		if(resizeCustomSe)
		{
			QSize sesize(mw->structuringElementSize());
			sesize = sesize * 2 + QSize(1,1);

			if(customSe.rows != sesize.height() ||
			   customSe.cols != sesize.width())
			{
				customSe = cv::Mat(sesize.height(), sesize.width(),
					CV_8UC1, cv::Scalar(0));
			}
		}

		emit structuringElementChanged(customSe);
	}	
}

void Controller::onStructuringElementPreviewPressed()
{
	static bool activated = false;
	static StructuringElementPreview* d;
	static QPoint pos(mw->pos() + QPoint(mw->geometry().width(), 0));

	if(!activated)
	{
		// Wygeneruj element strukturalny
		cv::Mat se = mw->structuringElementType() != Morphology::SET_Custom ?
			standardStructuringElement() : customSe;
		mw->setStructuringElementPreviewButtonText("Hide structuring element");

		d = new StructuringElementPreview(mw);
		d->setAttribute(Qt::WA_DeleteOnClose);
		d->move(pos);
		d->setModal(false);
		d->onStructuringElementChanged(se);
		d->show();

		connect(this, SIGNAL(structuringElementChanged(cv::Mat)), 
			d, SLOT(onStructuringElementChanged(cv::Mat)));
		connect(d, SIGNAL(closed()), 
			this, SLOT(onStructuringElementPreviewPressed()));

		activated = true;
	}
	else
	{
		// Wywolywane przez nacisniecie 'Hide..." LUB przez zamkniecie manualne okna dialogowego
		mw->setStructuringElementPreviewButtonText("Show structuring element");

		// Jesli jestesmy w sytuacji 1. to nalezy najpierw rozlaczyc signal - inaczej
		// close() wywola kolejnego closeEvent'a ktory stworzy na nowo okno
		disconnect(d, SIGNAL(closed()), 
			this, SLOT(onStructuringElementPreviewPressed()));

		pos = d->pos();
		activated = false;
		d->close();	
	}
}

void Controller::onStructuringElementModified(const cv::Mat& _customSe)
{
	customSe = _customSe;
	QSize sesize(customSe.cols, customSe.rows);
	sesize = (sesize - QSize(1,1)) / 2;

	resizeCustomSe = false;
	mw->setStructuringElementType(Morphology::SET_Custom);
	resizeCustomSe = true;
}

void Controller::onShowSourceImage()
{
	Q_ASSERT(mw->isNoneOperationChecked());

#if USE_GLWIDGET == 1
	// Pokaz obraz wynikowy
	if(useOpenCL && ocl->usingShared())
		previewGpuImage();
	else
#endif
		previewCpuImage(src);
}

void Controller::onRecompute()
{
	if(mw->isNoneOperationChecked())
		return;

	cv::Mat se = mw->structuringElementType() == Morphology::SET_Custom ?
		customSe : standardStructuringElement();
	Morphology::EOperationType op = mw->morphologyOperation();

	// Przetworz obraz
	if(!useOpenCL)
		processOpenCV(op, se);
	else
		processOpenCL(op, se);

#if USE_GLWIDGET == 1
	// Pokaz obraz wynikowy
	if(useOpenCL && ocl->usingShared())
		previewGpuImage();
	else
#endif
		previewCpuImage(dst);

	// Pozwol go zapisac (dla operacji None wylaczamy taka opcje)
	mw->allowImageSave();
}

void Controller::openFile(const QString& filename)
{
	src = cv::imread(filename.toStdString());
	
	int depth = src.depth();
	int channels = src.channels();

	qDebug("depth:%d channels:%d\n", depth, channels);

	Q_ASSERT(depth == CV_8U);

	if(channels == 3)
		cvtColor(src, src, CV_BGR2GRAY);
	else if(channels == 4)
		cvtColor(src, src, CV_BGRA2GRAY);

	setOpenCLSourceImage();
}

cv::Mat Controller::standardStructuringElement()
{
	using namespace Morphology;
	
	EStructuringElementType type = mw->structuringElementType();
	QSize elementSize = mw->structuringElementSize();
	int rotation = mw->structuringElementRotation();

	return Morphology::standardStructuringElement(
		elementSize.width(), elementSize.height(),
		type, rotation);
}

void Controller::showStats(int iters, double elapsed)
{
	QString txt;
	QTextStream strm(&txt);
	strm << "Time elapsed: " << elapsed << " ms, iterations: " << iters;
	printf("Time elapsed: %lf ms, iterations: %d\n", elapsed, iters);
	mw->setStatusBarText(txt);
}

void Controller::initializeOpenCL(EOpenCLMethod method)
{
	switch(method)
	{
	case OM_Buffer1D: ocl = new MorphOpenCLBuffer; break;
	case OM_Buffer2D: ocl = new MorphOpenCLImage; break;
	default: return;
	}

	ocl->setErrorCallback(
		[this](const QString& message, cl_int err)
	{
		Q_UNUSED(err);
		QMessageBox::critical(mw, "OpenCL critical error",
			QString("%1\nError code: %2").
				arg(message).
				arg(ocl->openCLErrorCodeStr(err)),
			QMessageBox::Ok);
		oclSupported = false;
	});

	oclSupported = true;
	oclSupported &= ocl->initialize();

	// A bit ugly :)
	ocl->setErrorCallback(
		[this](const QString& message, cl_int err)
	{
		Q_UNUSED(err);
		QMessageBox::critical(mw, "OpenCL critical error",
			QString("%1\nError code: %2").
			arg(message).
			arg(ocl->openCLErrorCodeStr(err)),
			QMessageBox::Ok);
	});
	useOpenCL = oclSupported;

	if(oclSupported)
	{
		mw->setOpenCLCheckableAndChecked(true);
	}
	else
	{
		QMessageBox::critical(nullptr,
			QLatin1String("OpenCL critical error"),
			QLatin1String("No OpenCL Platform available or something terrible happened "
				"during OpenCL initialization therefore OpenCL processing will be disabled."),
			QMessageBox::Ok);

		mw->setOpenCLCheckableAndChecked(false);
	}
}

void Controller::setOpenCLSourceImage()
{
	if(oclSupported)
	{
		if(ocl->usingShared())
		{
			GLuint glresource = previewWidget->createEmptySurface
				(src.cols, src.rows);
			ocl->setSourceImage(&src, glresource);
		}
		else
		{
			ocl->setSourceImage(&src);
		}
	}
}

void Controller::processOpenCV(Morphology::EOperationType op, const cv::Mat& se)
{
	cv::Mat src_(src);

	ElapsedTimer timer;
	timer.start();

	// Filtr Bayer'a
	if(mw->bayerIndex() != 0)
	{
		// Jest bug dla CV_BayerXX2GRAY i trzeba wykonac sciezke okrezna
		switch(mw->bayerIndex())
		{
		case 1: cv::cvtColor(src, src_, CV_BayerRG2BGR); break;
		case 2: cv::cvtColor(src, src_, CV_BayerGR2BGR); break;
		case 3: cv::cvtColor(src, src_, CV_BayerBG2BGR); break;
		case 4: cv::cvtColor(src, src_, CV_BayerGB2BGR); break;
		default: break;
		}
		cvtColor(src_, src_, CV_BGR2GRAY);
	}

	int iters = Morphology::process(src_, dst, op, se);
	double delapsed = timer.elapsed();

	// Wyswietl statystyki
	showStats(iters, delapsed);
}

void Controller::processOpenCL(Morphology::EOperationType op, const cv::Mat& se)
{
	ocl->error = false;
	int csize = ocl->setStructuringElement(se);

	if(ocl->error) 
		return;

	ocl->recompile(op, csize);

	if(ocl->error) 
		return;

	int iters;
	double delapsed = ocl->morphology(op, dst, iters);

	// Wyswietl statystyki
	showStats(iters, delapsed);
}

void Controller::previewCpuImage(const cv::Mat& image)
{
#if USE_GLWIDGET == 1
	QSize surfaceSize(image.cols, image.rows);
	double fx = CvUtil::scaleCoeff(
		cv::Size(conf.maxImageWidth, conf.maxImageHeight),
		image.size());

	surfaceSize.setWidth(surfaceSize.width() * fx);
	surfaceSize.setHeight(surfaceSize.height() * fx);

	previewWidget->setMinimumSize(surfaceSize);
	previewWidget->setMaximumSize(surfaceSize);
	previewWidget->setSurface(image);
#else
	cv::Mat img(image);
	cv::Size imgSize(conf.maxImageWidth, conf.maxImageHeight);
	CvUtil::resizeWithAspect(img, imgSize);
	QImage qimg(CvUtil::toQImage(img));
	previewLabel->setPixmap(QPixmap::fromImage(qimg));
#endif
}

void Controller::previewGpuImage()
{
	// TODO
}