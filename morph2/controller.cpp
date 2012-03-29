#include "controller.h"

#include <QFileDialog>
#include <QMessageBox>

#include "cvutils.h"
#include "elapsedtimer.h"
#include "settings.h"
#include "sepreview.h"

#include "procthread.h"
#include "capthread.h"

#define USE_GLWIDGET 0
#define DISABLE_OPENCL 1

Controller* Controller::msSingleton = nullptr;

Controller::Controller()
	: mw(nullptr)
	, ocl(nullptr)
	, negateSource(false)
	, oclSupported(false)
	, useOpenCL(false)
	, autoTrigger(false)
	, resizeCustomSe(true)
	, cameraConnected(false)
	, procQueue(3)
	, procThread(nullptr)
	, capThread(nullptr)
{
}

Controller::~Controller()
{
	// Zatrzymaj watki
	if(capThread && capThread->isRunning())
	{
		capThread->stop();
		capThread->wait();
		capThread->closeCamera();
		delete capThread;
	}

	if(procThread && procThread->isRunning())
	{
		procThread->stop();

		// W przypadku gdy kolejka zadan jest pusta watek by nie puscil
		if(procQueue.isEmpty())
		{
			ProcessingItem item = { false, cvu::BC_None,
				Morphology::OT_None, cv::Mat(), cv::Mat() };
			procQueue.enqueue(item);
		}

		procThread->wait();
		delete procThread;
	}

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

	// Utworz kontrolke podgladu obrazu
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

	// Inicjalizuj (sprobuj) OpenCLa
#if DISABLE_OPENCL == 0
	int method = 0;
	printf("There are 2 methods implemented:\n"
		"\t1) 2D buffer (image object)\n"
		"\t2) 1D buffer (buffer object)\n");
	while (method != 1 && method != 2)
	{
		printf("Choose method: ");
		int r = scanf("%d", &method);
		// Jesli nie odczytano jednej liczby (np. wprowadzono znak A)
		// trzeba oproznic stdin, inaczej wpadniemy w nieskonczona petle
		if(r != 1)
		{
			char buf[128];
			fgets(buf, 128, stdin);
		}
	}

	EOpenCLMethod emethod = static_cast<EOpenCLMethod>(method);
	initializeOpenCL(emethod);
#else
	mw->setOpenCLCheckableAndChecked(false);
#endif

	connect(mw, SIGNAL(recomputeNeeded()), SLOT(onRecompute()));
	connect(mw, SIGNAL(structuringElementChanged()), SLOT(onStructuringElementChanged()));

	// Odpal watek przetwarzajacy obraz
	procThread = new ProcThread(procQueue);

	// Customowe typy nalezy zarejstrowac dla polaczen typu QueuedConnection (miedzywatkowe)
	qRegisterMetaType<ProcessedItem>("ProcessedItem");
	connect(procThread, SIGNAL(processingDone(ProcessedItem)), SLOT(onProcessingDone(ProcessedItem)));
	
	procThread->start(QThread::HighPriority);

	openFile(defImage);
	onRecompute();

	mw->setCameraStatusBarState(false);
	mw->show();
}

void Controller::onFromCameraTriggered(bool state)
{
	if(state)
	{
		capThread = new CapThread(procQueue);
		if(!(cameraConnected = capThread->openCamera(0)))
		{
			mw->setFromCamera(false);

			QMessageBox::critical(mw, "Error", 
				"Cannot establish connection to selected camera device.", 
				QMessageBox::Ok);
		}
		else
		{
			Morphology::EOperationType op = mw->morphologyOperation();
			cvu::EBayerCode bc = static_cast<cvu::EBayerCode>(mw->bayerIndex());
			cv::Mat se = structuringElement();

			capThread->setJobDescription(negateSource, bc, op, se);
			capThread->start(QThread::LowPriority);

			mw->setEnabledSaveOpenFile(false);
		}
	}
	else
	{
		if(capThread->isRunning())
		{
			capThread->stop();
			capThread->wait();
		}

		// src = [Ostatnia klatka z kamery czy zostajemy przy ostatnim wczytanym obrazie z dysku]
		src = capThread->currentFrame().clone();
		capThread->closeCamera();

		delete capThread;
		capThread = 0;
		cameraConnected = false;

		mw->setEnabledSaveOpenFile(true);
	}

	mw->setCameraStatusBarState(cameraConnected);
}

void Controller::onOpenFileTriggered()
{
	QString filename = QFileDialog::getOpenFileName(mw, QString(), ".",
		QLatin1String("Image files (*.png *.jpg *.bmp)"));

	if(filename.isEmpty())
		return;

	openFile(filename);

	if(mw->morphologyOperation() != Morphology::OT_None)
		// wywola onRecompute()
		mw->setMorphologyOperation(Morphology::OT_None); 
	else
		// jesli wybrane wczesniej bylo None to nie zostal
		// wyemitowany zaden sygnal
		onRecompute();

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

	// Serializuje magic'a oraz typ elementu strukturalnego
	auto type = mw->structuringElementType();
	strm << 0x1337U << type;

	if(type != Morphology::SET_Custom)
	{
		// Serializuj parametry SE
		QSize ksize = mw->structuringElementSize();
		int rotation = mw->structuringElementRotation();

		strm << ksize << rotation;
	}
	else
	{
		// Serializuj wszystkie wartosci pikseli
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
	negateSource = state;

	if(!cameraConnected && 
		(autoTrigger || mw->isNoneOperationChecked()))
		onRecompute();
	else if(cameraConnected)
		if(capThread)
			capThread->setNegateImage(state);
}

void Controller::onOpenCLTriggered(bool state)
{
	useOpenCL = state;

	if(!cameraConnected && autoTrigger && 
		!mw->isNoneOperationChecked())
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

	// Jaki "silnik" wybrano
	if(msgBox.clickedButton() == buffer1D)
		method = OM_Buffer1D;
	else if(msgBox.clickedButton() == buffer2D)
		method = OM_Buffer2D;
	else
		return;

	// Reinicjalizuj modul OpenCLa z wybranym silnikiem
	delete ocl;
	initializeOpenCL(method);
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
		if(!cameraConnected)
			onRecompute();
	}
	else
	{
		autoTrigger = false;
	}
}

void Controller::onBayerIndexChanged(int bcode)
{
	cvu::EBayerCode bc = static_cast<cvu::EBayerCode>(bcode);

	if(oclSupported)
		ocl->setBayerFilter(bc);

	// Dla no-op rowniez chcemy to wykonac
	if(!cameraConnected && (autoTrigger || cameraConnected))
		onRecompute();
	else if(cameraConnected)
		if(capThread)
			capThread->setBayerCode(bc);
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

void Controller::onRecompute()
{
	Morphology::EOperationType op = mw->morphologyOperation();
	cvu::EBayerCode bc = static_cast<cvu::EBayerCode>(mw->bayerIndex());
	cv::Mat se = structuringElement();

	if(cameraConnected)
	{
		if(capThread)
			capThread->setJobDescription(negateSource, bc, op, se);
		return;
	}

	//if(!useOpenCL)
	// procQueue.enqueue(item);
	//else
	// procOpenCLQueue.enqueue(item);

	ProcessingItem item = { negateSource, bc, op, se, src };
	procQueue.enqueue(item);
}

void Controller::onProcessingDone(const ProcessedItem& item)
{
	//#if USE_GLWIDGET == 1
	//	// Pokaz obraz wynikowy
	//	if(useOpenCL && ocl->usingShared())
	//		previewGpuImage();
	//	else
	//#endif
	//		previewCpuImage(dst);

	//	void Controller::onShowSourceImage()
	//	{
	//		Q_ASSERT(mw->isNoneOperationChecked());
	//
	//#if USE_GLWIDGET == 1
	//		// Pokaz obraz wynikowy
	//		if(useOpenCL && ocl->usingShared())
	//			previewGpuImage();
	//		else
	//#endif
	//			previewCpuImage(src);
	//	}

	dst = item.dst;
	previewCpuImage(dst);
	showStats(item.iters, item.delapsed);
}

void Controller::openFile(const QString& filename)
{
	src = cv::imread(filename.toStdString());
	
	int depth = src.depth();
	int channels = src.channels();

	Q_ASSERT(depth == CV_8U);

	if(channels == 3)
		cvtColor(src, src, CV_BGR2GRAY);
	else if(channels == 4)
		cvtColor(src, src, CV_BGRA2GRAY);
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

cv::Mat Controller::structuringElement()
{
	Morphology::EOperationType op = mw->morphologyOperation();
	return (op == Morphology::OT_None) ? cv::Mat() : 
		((mw->structuringElementType() == Morphology::SET_Custom) ?
			customSe : standardStructuringElement());
}

void Controller::showStats(int iters, double elapsed)
{
	QString txt;
	QTextStream strm(&txt);
	strm << "Time elapsed: " << elapsed << " ms, iterations: " << iters;
	printf("Time elapsed: %lf ms, iterations: %d\n", elapsed, iters);
	mw->setStatusBarText(txt);
}

void Controller::previewCpuImage(const cv::Mat& image)
{
#if USE_GLWIDGET == 1
	QSize surfaceSize(image.cols, image.rows);
	double fx = cvu::scaleCoeff(
		cv::Size(conf.maxImageWidth, conf.maxImageHeight),
		image.size());

	surfaceSize.setWidth(surfaceSize.width() * fx);
	surfaceSize.setHeight(surfaceSize.height() * fx);

	previewWidget->setMinimumSize(surfaceSize);
	previewWidget->setMaximumSize(surfaceSize);
	previewWidget->setSurface(image);
#else
	static const cv::Size imgSize(conf.maxImageWidth, conf.maxImageHeight);

	cv::Mat img(image);
	double fx = cvu::scaleCoeff(imgSize, image.size());
	cv::resize(img, img, cv::Size(), fx, fx, cv::INTER_LINEAR);
	//cvu::resizeWithAspect(img, imgSize);

	// Konwersja cv::Mat -> QImage -> QPixmap
	QImage qimg(cvu::toQImage(img));
	previewLabel->setPixmap(QPixmap::fromImage(qimg));
#endif
}

////////////////////////////////////////////////////////////////////////////////

void Controller::initializeOpenCL(EOpenCLMethod method)
{
	// Inicjalizuj odpowiedni silnik
	switch(method)
	{
	case OM_Buffer1D: ocl = new MorphOpenCLBuffer(conf); break;
	case OM_Buffer2D: ocl = new MorphOpenCLImage(conf); break;
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

	// Troche paskudne rozwiazanie :)
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

//void Controller::setOpenCLSourceImage()
//{
//	if(oclSupported)
//	{
//		if(ocl->usingShared())
//		{
//			GLuint glresource = previewWidget->createEmptySurface
//				(src.cols, src.rows);
//			ocl->setSourceImage(&src, glresource);
//		}
//		else
//		{
//			ocl->setSourceImage(&src);
//		}
//	}
//}
//
//void Controller::processOpenCL(Morphology::EOperationType op, const cv::Mat& se)
//{
//	ocl->error = false;
//	int csize = ocl->setStructuringElement(se);
//
//	if(ocl->error) 
//		return;
//
//	ocl->recompile(op, csize);
//
//	if(ocl->error) 
//		return;
//
//	int iters;
//	double delapsed = ocl->morphology(op, dst, iters);
//
//	// Wyswietl statystyki
//	showStats(iters, delapsed);
//}


//void Controller::previewGpuImage()
//{
//#if USE_GLWIDGET == 1
//	// Mozna by przy wczytywaniu ustawic te wielkosci
//
//	//QSize surfaceSize(src.cols, src.rows);
//	//double fx = CvUtil::scaleCoeff(
//	//	cv::Size(conf.maxImageWidth, conf.maxImageHeight),
//	//	image.size());
//	//surfaceSize.setWidth(surfaceSize.width() * fx);
//	//surfaceSize.setHeight(surfaceSize.height() * fx);
//	//previewWidget->setMinimumSize(surfaceSize);
//	//previewWidget->setMaximumSize(surfaceSize);
//
//	previewWidget->updateGL();
//#endif
//}