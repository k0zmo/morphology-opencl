#include "controller.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QProgressBar>

#include "cvutils.h"
#include "elapsedtimer.h"
#include "settings.h"
#include "sepreview.h"
#include "oclpicker.h"

#include "procthread.h"
#include "oclthread.h"
#include "capthread.h"

#define USE_GLWIDGET 1
#define DISABLE_OPENCL 1

template<>
Controller* Singleton<Controller>::msSingleton = nullptr;

Controller::Controller(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
	, mw(nullptr)
	, preview(nullptr)
	, negateSource(false)
	, oclSupported(false)
	, useOpenCL(false)
	, autoTrigger(false)
	, resizeCustomSe(true)
	, cameraConnected(false)
	, procQueue(3)
	, clQueue(3)
	, procThread(nullptr)
	, capThread(nullptr)
	, clThread(nullptr)
{
	setupUi(this);

	// Menu (File)
	connect(actionCameraInput, SIGNAL(triggered(bool)), SLOT(onFromCameraTriggered(bool)));
	connect(actionOpen, SIGNAL(triggered()), SLOT(onOpenFileTriggered()));
	connect(actionSave, SIGNAL(triggered()), SLOT(onSaveFileTriggered()));
	connect(actionOpenSE, SIGNAL(triggered()), SLOT(onOpenStructuringElementTriggered()));
	connect(actionSaveSE, SIGNAL(triggered()), SLOT(onSaveStructuringElementTriggered()));
	connect(actionExit, SIGNAL(triggered()), SLOT(close()));

	// Menu (Settings)
	connect(actionOpenCL, SIGNAL(triggered(bool)), SLOT(onOpenCLTriggered(bool)));
	connect(actionPickMethod, SIGNAL(triggered()), SLOT(onPickMethodTriggerd()));
	connect(actionSettings, SIGNAL(triggered()), SLOT(onSettingsTriggered()));

	// Pasek stanu
	procQueueLabel = new QLabel(this);
	statusBar()->addPermanentWidget(procQueueLabel);

	statusBarLabel = new QLabel(this);
	statusBar()->addPermanentWidget(statusBarLabel);

	cameraStatusLabel = new QLabel(this);
	statusBar()->addPermanentWidget(cameraStatusLabel);
	setCameraStatusBarState(false);
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
				cvu::MO_None, cv::Mat(), cv::Mat() };
			procQueue.enqueue(item);
		}

		procThread->wait();
		delete procThread;
	}

	if(clThread && clThread->isRunning())
	{
		clThread->stop();

		// W przypadku gdy kolejka zadan jest pusta watek by nie puscil
		if(clQueue.isEmpty())
		{
			ProcessingItem item = { false, cvu::BC_None,
				cvu::MO_None, cv::Mat(), cv::Mat() };
			clQueue.enqueue(item);
		}

		clThread->wait();
		delete clThread;
	}

	delete mw;
}

void Controller::start()
{
	conf.loadConfiguration("./settings.cfg");
	QString defImage(conf.defaultImage);

	// Wczytaj domyslny obraz (jesli wyspecyfikowano)
	if(defImage.isEmpty())
	{
		defImage = QFileDialog::getOpenFileName(this, QString(), ".",
			QLatin1String("Image files (*.png *.jpg *.bmp)"));

		if(defImage.isEmpty())
			// TODO mozna by start dac jako SLOT i wtedy wywolac close()
			// Konstruktor wywolalby zdarzenie ktore juz byloby przetworzone
			// w petli komunikatow
			exit(-1);
	}

	openFile(defImage);

	mw = new MainWidget(this);
	connect(mw, SIGNAL(recomputeNeeded()), SLOT(onRecompute()));
	connect(mw, SIGNAL(structuringElementChanged()), SLOT(onStructuringElementChanged()));

	gridLayout->addWidget(mw, 0, 1, 1, 1);

	// Utworz kontrolke do podgladu obrazu
	preview = new PreviewProxy(this);
	gridLayout->addWidget(preview, 0, 0, 1, 1);

	QSpacerItem* spacer = new QSpacerItem(0, 0,
		QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);
	gridLayout->addItem(spacer, 1, 0, 1, 2);

	// Uruchom procedure inicjalizacyjna okno podgladu
	// gdy bedzie gotowe - uruchomi cala reszte
	connect(preview, SIGNAL(initialized(bool)), SLOT(onPreviewInitialized(bool)));
#if USE_GLWIDGET == 0
	preview->initSoftware();
#else
	preview->initHardware();
#endif
}

void Controller::onPreviewInitialized(bool success)
{
	Q_UNUSED(success)

	// Customowe typy nalezy zarejstrowac dla polaczen
	// typu QueuedConnection (miedzywatkowe)
	qRegisterMetaType<ProcessedItem>("ProcessedItem");

	// Odpal watek przetwarzajacy obraz
	procThread = new ProcThread(procQueue);
	connect(procThread, SIGNAL(processingDone(ProcessedItem)),
		SLOT(onProcessingDone(ProcessedItem)));
	procThread->start(QThread::HighPriority);

#if DISABLE_OPENCL == 0
	initializeOpenCL();
#endif

	onRecompute();
	show();
}

void Controller::onRecompute()
{
	// Pobierz parametry operacji
	cvu::EMorphOperation op = mw->morphologyOperation();
	cvu::EBayerCode bc = static_cast<cvu::EBayerCode>(mw->bayerIndex());
	cv::Mat se = structuringElement();

	if(cameraConnected)
	{
		if(capThread)
			capThread->setJobDescription(negateSource, bc, op, se);
		return;
	}

	// Utworz 'obiekt przetwarzany'
	ProcessingItem item = { negateSource, bc, op, se, src };

	if(useOpenCL)
		clQueue.enqueue(item);
	else
		procQueue.enqueue(item);

	setEnqueueJobsStatus();
}

void Controller::onProcessingDone(const ProcessedItem& item)
{
	setEnqueueJobsStatus();
	dst = item.dst;

	QSize maxImgSize(conf.maxImageWidth, conf.maxImageHeight);
	preview->setPreviewImage(dst, maxImgSize);

	showStats(item.iters, item.delapsed);
}

void Controller::onFromCameraTriggered(bool state)
{
	mw->setCameraStatus(state);

	if(state)
	{
		capThread = new CapThread(useOpenCL ? 1 : 0, procQueue, clQueue);
		if(!(cameraConnected = capThread->openCamera(0)))
		{
			// Wracamy do stanu sprzed wybrania kamery jako wejscia
			actionCameraInput->setChecked(false);
			mw->setCameraStatus(false);

			QMessageBox::critical(mw, "Error", 
				"Cannot establish connection to selected camera device.", 
				QMessageBox::Ok);
		}
		else
		{
			cvu::EMorphOperation op = mw->morphologyOperation();
			cvu::EBayerCode bc = static_cast<cvu::EBayerCode>(mw->bayerIndex());
			cv::Mat se = structuringElement();

			capThread->setJobDescription(negateSource, bc, op, se);
			capThread->start(QThread::LowPriority);

			setEnabledSaveOpenFile(false);
			mw->setEnabledAutoRecompute(false);
		}
	}
	else
	{
		// Zakoncz dzialanie watku kamerki
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

		setEnabledSaveOpenFile(true);
		mw->setEnabledAutoRecompute(true);
	}

	setCameraStatusBarState(cameraConnected);
}

void Controller::onOpenFileTriggered()
{
	QString filename = QFileDialog::getOpenFileName(this, QString(), ".",
		QLatin1String("Image files (*.png *.jpg *.bmp)"));

	if(filename.isEmpty())
		return;

	openFile(filename);

	if(mw->morphologyOperation() != cvu::MO_None)
		// wywola onRecompute()
		mw->setMorphologyOperation(cvu::MO_None); 
	else
		// jesli wybrane wczesniej bylo None to nie zostal
		// wyemitowany zaden sygnal
		onRecompute();

	mw->resize(1, 1);
}

void Controller::onSaveFileTriggered()
{
	QString filename = QFileDialog::getSaveFileName(this, QString(), ".",
		QLatin1String("Image file (*.png)"));

	if(filename.isEmpty())
		return;

	cv::Mat dstc; cvtColor(dst, dstc, CV_GRAY2BGR);
	cv::imwrite(filename.toStdString(), dstc);
}

void Controller::onOpenStructuringElementTriggered()
{
	QString filename = QFileDialog::getOpenFileName(this, QString(), ".",
		QLatin1String("Structuring element file (*.se)"));
	
	if(filename.isEmpty())
		return;

	// Dane do deserializacji
	cvu::EStructuringElementType etype;
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
	etype = static_cast<cvu::EStructuringElementType>(type);

	if(etype != cvu::SET_Custom)
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
	QString filename = QFileDialog::getSaveFileName(this, QString(), ".",
		QLatin1String("Structuring element file (*.se)"));

	if(filename.isEmpty())
		return;

	QFile file(filename);
	file.open(QIODevice::WriteOnly);
	QDataStream strm(&file);

	// Serializuje magic'a oraz typ elementu strukturalnego
	auto type = mw->structuringElementType();
	strm << 0x1337U << type;

	if(type != cvu::SET_Custom)
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

	// Wyczysc poprzednie kolejki
	clQueue.clear();
	procQueue.clear();

	if(!cameraConnected && autoTrigger && 
		!mw->isNoneOperationChecked())
		onRecompute();

	if(capThread)
		capThread->setUsedQueue(useOpenCL ? 1 : 0);
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

//	EOpenCLMethod method;

//	// Jaki "silnik" wybrano
//	if(msgBox.clickedButton() == buffer1D)
//		method = OM_Buffer1D;
//	else if(msgBox.clickedButton() == buffer2D)
//		method = OM_Buffer2D;
//	else
//		return;

	// Reinicjalizuj modul OpenCLa z wybranym silnikiem
//	delete ocl;
//	initializeOpenCL(method);
}

void Controller::onSettingsTriggered()
{
	Settings s;
	s.setConfigurationModel(conf);

	int ret = s.exec();
	if(ret != QDialog::Accepted)
		return;

	Configuration conf = s.configurationModel();
	conf.saveConfiguration("./settings.cfg");		

	QMessageBox::information(this, "Settings",
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

	// Dla no-op rowniez chcemy to wykonac
	if(!cameraConnected && (autoTrigger || cameraConnected))
		onRecompute();
	else if(cameraConnected)
		if(capThread)
			capThread->setBayerCode(bc);
}

void Controller::onStructuringElementChanged()
{
	if(mw->structuringElementType() != cvu::SET_Custom)
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
	static QPoint position(pos() + QPoint(geometry().width(), 0));

	if(!activated)
	{
		// Wygeneruj element strukturalny
		cv::Mat se = mw->structuringElementType() != cvu::SET_Custom ?
			standardStructuringElement() : customSe;
		mw->setStructuringElementPreviewButtonText("Hide structuring element");

		d = new StructuringElementPreview(this);
		d->setAttribute(Qt::WA_DeleteOnClose);
		d->move(position);
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

		position = d->pos();
		activated = false;
		d->close();	
		d->deleteLater();
	}
}

void Controller::onStructuringElementModified(const cv::Mat& _customSe)
{
	customSe = _customSe;
	QSize sesize(customSe.cols, customSe.rows);
	sesize = (sesize - QSize(1,1)) / 2;

	resizeCustomSe = false;
	mw->setStructuringElementType(cvu::SET_Custom);
	resizeCustomSe = true;
}

////////////////////////////////////////////////////////////////////////////////

void Controller::openFile(const QString& filename)
{
	src = cv::imread(filename.toStdString());
	
	int depth = src.depth();
	int channels = src.channels();

	Q_ASSERT(depth == CV_8U);
	Q_UNUSED(depth);
	Q_UNUSED(channels);

	if(channels == 3)
		cvtColor(src, src, CV_BGR2GRAY);
	else if(channels == 4)
		cvtColor(src, src, CV_BGRA2GRAY);
}

cv::Mat Controller::standardStructuringElement()
{
	using namespace cvu;
	
	EStructuringElementType type = mw->structuringElementType();
	QSize elementSize = mw->structuringElementSize();
	int rotation = mw->structuringElementRotation();

	return cvu::standardStructuringElement(
		elementSize.width(), elementSize.height(),
		type, rotation);
}

cv::Mat Controller::structuringElement()
{
	cvu::EMorphOperation op = mw->morphologyOperation();
	return (op == cvu::MO_None) ? cv::Mat() : 
		((mw->structuringElementType() == cvu::SET_Custom) ?
			customSe : standardStructuringElement());
}

void Controller::showStats(int iters, double elapsed)
{
	QString txt;
	QTextStream strm(&txt);
	strm << "Time elapsed: " << elapsed << " ms, iterations: " << iters;
	printf("Time elapsed: %lf ms, iterations: %d\n", elapsed, iters);
	statusBarLabel->setText(txt);
}

////////////////////////////////////////////////////////////////////////////////

void Controller::initializeOpenCL()
{
	// Odpal watek przetwarzajacy obraz z wykorzystaniem OpenCLa
	clThread = new oclThread(clQueue, conf);
	connect(clThread, SIGNAL(processingDone(ProcessedItem)),
		SLOT(onProcessingDone(ProcessedItem)));
	connect(clThread, SIGNAL(openCLInitialized(bool)),
		SLOT(onOpenCLInitialized(bool)));

	PlatformDevicesMap map = clThread->queryPlatforms();
	oclPicker picker(map);

	// Wartosci domyslne
	int platformId = 0;
	int deviceId = 0;

	if(picker.exec() == QDialog::Accepted)
	{
		platformId = picker.platform();
		deviceId = picker.device();
	}

	clThread->choose(platformId, deviceId);
	clThread->start(QThread::HighPriority);

	statusBarLabel->setText("Initializing OpenCL context...");

	QProgressBar* pb = new QProgressBar(qstatusBar);
	pb->setObjectName("ProgressBar");
	pb->setRange(0, 0);
	qstatusBar->insertWidget(2, pb);
}

void Controller::onOpenCLInitialized(bool success)
{
	// Progress bar najpierw
	QProgressBar* pb = qstatusBar->findChild<QProgressBar*>("ProgressBar");
	if(!pb) return;
	qstatusBar->removeWidget(pb);
	pb->deleteLater();

	if(success)
	{
		qDebug("OpenCL Context intialized successfully\n");

		statusBarLabel->setText("OpenCL Context intialized successfully");
		setOpenCLCheckableAndChecked(true);
		useOpenCL = true;
	}
	else
	{
		QMessageBox::critical(nullptr, "Morph OpenCL",
			"Error occured during OpenCL Context initialization.\n"
			"Check console for more detailed description.\n"
			"OpenCL processing will be disabled now.");

		statusBarLabel->setText("Error occured during OpenCL Context initialization.");
		setOpenCLCheckableAndChecked(false);
		useOpenCL = false;
		qDebug("\n");
	}
}

