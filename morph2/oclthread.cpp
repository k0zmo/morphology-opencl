#include "glwidget.h"

#include <Qt/QtOpenCL>

//#include "oclcontext.h"
//#include "oclfilter.h"
#include "oclbayerfilter.h"
//#include "oclmorphfilter.h"
//#include "oclmorphhitmissfilter.h"

#include "oclthread.h"
#include "cvutils.h"
#include "elapsedtimer.h"

#include <QDebug>

#ifdef Q_WS_X11
#	include <GL/glx.h>
#endif

qreal eventDuration(const QCLEvent& evt)
{
	qreal runDuration = (evt.finishTime() - evt.runTime()) / 1000000.0f;
	return runDuration;
}

oclThread::oclThread(BlockingQueue<ProcessingItem>& queue,
	const Configuration& conf, GLDummyWidget* shareWidget)
	: QThread(nullptr)
	, queue(queue)
	, shareWidget(shareWidget)
	, platformId(0)
	, deviceId(0)
	, stopped(false)
	, success(true)
	, conf(conf)
{
}

oclThread::~oclThread()
{
}

void oclThread::stop()
{
	QMutexLocker locker(&stopMutex);
	stopped = true;
}

PlatformDevicesMap oclThread::queryPlatforms()
{
	PlatformDevicesMap plToDevs;

	QList<QCLPlatform> pls = QCLPlatform::platforms();
	foreach(QCLPlatform pl, pls)
	{
		QList<QCLDevice> devs = QCLDevice::devices(QCLDevice::All, pl);
		plToDevs.insert(pl, devs);
	}

	return plToDevs;
}

void oclThread::initContext(QCLContext& ctx)
{
	// TODO: reuse from queryPlatforms
	QList<QCLPlatform> pls = QCLPlatform::platforms();

	if(platformId >= pls.count())
	{
		success = false;
		return;
	}

	QCLPlatform pl = pls[platformId];
	QList<QCLDevice> devs = QCLDevice::devices(QCLDevice::All, pl);

	if(deviceId >= devs.count())
	{
		success = false;
		return;
	}

	QCLDevice dev = devs[deviceId];
	QList<QCLDevice> devList;
	devList << dev;
	ctx.create(devList);

	QCLCommandQueue cq = ctx.createCommandQueue(CL_QUEUE_PROFILING_ENABLE);
	ctx.setCommandQueue(cq);

//	if(shareWidget)
//	{
//		// createContextGL oczekuje aktywnego kontekstu
//		// Moze pozniej to zmienie (bedzie oczekiwac tych wartosci w argumentach funkcji)
//		qDebug() << "(i) Przed makeCurrent";
//		shareWidget->makeCurrent();
//		qDebug() << "(i) Po makeCurrent";

//#ifdef Q_WS_WIN32
//		HDC dc = wglGetCurrentDC();
//		HGLRC rc = wglGetCurrentContext();
//#else
//		Display* dc = glXGetCurrentDisplay();
//		GLXContext rc = glXGetCurrentContext();
//#endif

//		//qDebug() << "currentDC:" << dc <<
//		//	"currentContext:" << rc;

//		// TODO: for glx
//		// TODO: albo przeniesc to do createContextGL
//		if(!dc || !rc)
//		{
//			c.createContext(platformId);
//			shareWidget = nullptr;
//		}
//		else
//		{
//			c.createContextGL(platformId);

//			qDebug() << "(i) Przed doneCurrent";
//			shareWidget->doneCurrent();
//			qDebug() << "(i) Po doneCurrent";
//		}
//	}
//	else
//	{
//		c.createContext(platformId);
//	}

//	// wybor urzadzenia
//	c.chooseDevice(deviceId);

//	c.createCommandQueue(true);
//	c.setWorkgroupSize(conf.workgroupSizeX,
//		conf.workgroupSizeY);

//	// Bloczek z filtracja bayera
//	bayerFilter = new oclBayerFilter(&c);

//	// Bloczek z operacjami morfologicznymi
//	morphFilter = new oclMorphFilter(&c,
//		conf.erode_2d.toStdString().c_str(),
//		conf.dilate_2d.toStdString().c_str(),
//		conf.gradient_2d.toStdString().c_str());

//	// Bloczek z operacjami morfologicznymi typu hit-miss
//	hitmissFilter = new oclMorphHitMissFilter(&c,
//		conf.atomicCounters);
}

void oclThread::choose(int platformId_, int deviceId_)
{
	platformId = platformId_;
	deviceId = deviceId_;
}

void oclThread::run()
{
	QCLContext ctx;
	initContext(ctx);

	oclBayerFilter bayerFilter(&ctx);
	success = ctx.lastError() == CL_SUCCESS;

	emit openCLInitialized(success);
	if(!success)
		return;

	while(true)
	{
		ProcessingItem item = queue.dequeue();

		{
			QMutexLocker locker(&stopMutex);
			if(stopped)
				break;
		}

		#ifdef _DEBUG
		qDebug() << endl << "New processing job (OpenCL):" << "\n\toperation:" <<
			item.op << "\n\tbayer code:" << item.bc <<
			"\n\tnegate:" << item.negate << endl;
		#endif

		// Brak operacji, zwroc obraz zrodlowy
		if(!item.negate &&
			item.bc == cvu::BC_None &&
			item.op == cvu::MO_None)
		{
			ProcessedItem pitem = {
				/*.iters = */ 0,
				/*.delapsed = */ 0.0,
				/*.dst = */ item.src,
				/*.glsize = */ cv::Size(0, 0) // <- bez interopu
			};
			emit processingDone(pitem);
			continue;
		}

		ProcessedItem pitem = {
			/*.iters = */ 1,
			/*.delapsed = */ 0.0,
			/*.dst = */ cv::Mat(),
			/*glsize = */ cv::Size(0, 0)
		};

		// Zaneguj obraz
		// TODO: Bloczek do negowania
		if(item.negate)
		{
			ElapsedTimer t;
			t.start();

			cv::Mat tmp(item.src.size(), item.src.depth(), item.src.channels());
			cvu::negate(item.src, tmp);
			item.src = tmp;

			pitem.delapsed += t.elapsed();

			// Czy jest to ostatni 'bloczek'
			if (item.bc == cvu::BC_None &&
				item.op == cvu::MO_None)
			{
				pitem.dst = item.src;
				emit processingDone(pitem);
				continue;
			}
		}

		// Utworz obraz na urzadzeniu
		QCLImage2D holder = ctx.createImage2DDevice
				(morphImageFormat(), QSize(item.src.cols, item.src.rows),
				 QCLMemoryObject::ReadOnly);

		// Skopiuj do niego dane
		QCLEvent evt = holder.writeAsync
				(const_cast<uchar*>(item.src.ptr<uchar>()),
				 QRect(0, 0, item.src.cols, item.src.rows),
				 QCLEventList(), 0);
		evt.waitForFinished();

		//qDebug() << evt;

		// Tego tez mozemy liczyc czas
		qreal elapsed = eventDuration(evt);
		qDebug("\nTransfering source image to the device took %.05lf ms", elapsed);
		pitem.delapsed += elapsed;

		// Filtr Bayer'a
		//if(item.bc != cvu::BC_None)
		{
			bayerFilter.setBayerFilter(cvu::BC_None /*item.bc*/);
			bayerFilter.setSourceImage(holder);

//			// Czy filtr bayera jest ostatnim 'bloczkiem'
//			if(glInterop && item.op == cvu::MO_None)
//			{
//				oclImage2DHolder output = c.createDeviceImageGL(shareWidget->surface(), WriteOnly);
//				bayerFilter->setOutputDeviceImage(output);
//				pitem.delapsed += bayerFilter->run();
//			}
//			else
			{
				QCLEvent evt = bayerFilter.run();
				pitem.delapsed += eventDuration(evt);
				holder = bayerFilter.outputDeviceImage();
			}
		}

//		if(!glInterop)
		{
			// Jesli nie dzielimy zasobow, trzeba je teraz sciagnac

			// TODO:
			int format = CV_8U;
			pitem.dst = cv::Mat(cv::Size(holder.width(), holder.height()),
					format, cv::Scalar(1));

			QCLEvent evt = holder.readAsync
					(pitem.dst.ptr<uchar>(),
					 QRect(0, 0, pitem.dst.cols, pitem.dst.rows),
					 QCLEventList(), 0);
			evt.waitForFinished();

			qreal elapsed = eventDuration(evt);
			qDebug("Transfering output image from the device took %.05lf ms", elapsed);
			pitem.delapsed += elapsed;
		}
//		else
//		{
//			shareWidget->doneCurrent();
//		}

		emit processingDone(pitem);
	}

//	// Czasem sie zdarza ze implementacja CPU zglasza obsluge cl/gl interop
//	// Jednak nie jest to prawda :)
//	bool glInterop = shareWidget != nullptr;
//	oclDeviceDesc ddesc = c.deviceDescription();
//	if(ddesc.deviceType != CL_DEVICE_TYPE_GPU)
//		glInterop = false;

//	printf("GL/CL Interop: %s\n", glInterop
//		? "yes"
//		: "no");

//	while(true)
//	{
//		ProcessingItem item = queue.dequeue();

//		{
//			QMutexLocker locker(&stopMutex);
//			if(stopped)
//				break;
//		}

//		#ifdef _DEBUG
//		qDebug() << endl << "New processing job (OpenCL):" << "\n\toperation:" <<
//			item.op << "\n\tbayer code:" << item.bc <<
//			"\n\tnegate:" << item.negate << endl;
//		#endif

//		// Brak operacji, zwroc obraz zrodlowy
//		if(!item.negate &&
//			item.bc == cvu::BC_None &&
//			item.op == cvu::MO_None)
//		{
//			ProcessedItem pitem = {
//				/*.iters = */ 0,
//				/*.delapsed = */ 0.0,
//				/*.dst = */ item.src,
//				/*.glsize = */ cv::Size(0, 0) // <- bez interopu
//			};
//			emit processingDone(pitem);
//			continue;
//		}

//		ProcessedItem pitem = {
//			/*.iters = */ 1,
//			/*.delapsed = */ 0.0,
//			/*.dst = */ cv::Mat(),
//			/*glsize = */ cv::Size(0, 0)
//		};

//		// Zaneguj obraz
//		// TODO: Bloczek do negowania
//		if(item.negate)
//		{
//			ElapsedTimer t;
//			t.start();

//			cv::Mat tmp(item.src.size(), item.src.depth(), item.src.channels());
//			cvu::negate(item.src, tmp);
//			item.src = tmp;
			
//			pitem.delapsed += t.elapsed();

//			// Czy jest to ostatni 'bloczek'
//			if (item.bc == cvu::BC_None &&
//				item.op == cvu::MO_None)
//			{
//				pitem.dst = item.src;
//				emit processingDone(pitem);
//				continue;
//			}
//		}

//		if(glInterop)
//		{
//			pitem.glsize = item.src.size();
//			shareWidget->makeCurrent();
//			shareWidget->resizeSurface(pitem.glsize.width, pitem.glsize.height);
//		}
		
//		oclImage2DHolder holder = c.copyImageToDevice(item.src, ReadOnly);

//		// Tego tez mozemy liczyc czas
//		qDebug("Transfering source image to the device took %.05lf ms\n",
//			c.oclElapsedEvent(holder.evt));
//		pitem.delapsed += c.oclElapsedEvent(holder.evt);
			
//		// Filtr Bayer'a
//		if(item.bc != cvu::BC_None)
//		{
//			bayerFilter->setBayerFilter(item.bc);
//			bayerFilter->setSourceImage(holder);

//			// Czy filtr bayera jest ostatnim 'bloczkiem'
//			if(glInterop && item.op == cvu::MO_None)
//			{
//				oclImage2DHolder output = c.createDeviceImageGL(shareWidget->surface(), WriteOnly);
//				bayerFilter->setOutputDeviceImage(output);
//				pitem.delapsed += bayerFilter->run();
//			}
//			else
//			{
//				pitem.delapsed += bayerFilter->run();
//				holder = oclImage2DHolder(bayerFilter->outputDeviceImage());
//			}
//		}

//		// Operacja morfologiczna/hitmiss
//		if(item.op != cvu::MO_None)
//		{
//			if(isHitMiss(item.op))
//			{
//				hitmissFilter->setHitMissOperation(item.op);
//				hitmissFilter->setSourceImage(holder);

//				if(glInterop)
//				{
//					oclImage2DHolder output = c.createDeviceImageGL(shareWidget->surface(), WriteOnly);
//					hitmissFilter->setOutputDeviceImage(output);
//					pitem.delapsed += hitmissFilter->run();
//				}
//				else
//				{
//					pitem.delapsed += hitmissFilter->run();
//					holder = oclImage2DHolder(morphFilter->outputDeviceImage());
//				}
//			}
//			else
//			{
//				morphFilter->setMorphologyOperation(item.op);
//				morphFilter->setStructuringElement(item.se);
//				morphFilter->setSourceImage(holder);

//				if(glInterop)
//				{
//					oclImage2DHolder output = c.createDeviceImageGL(shareWidget->surface(), WriteOnly);
//					morphFilter->setOutputDeviceImage(output);
//					pitem.delapsed += morphFilter->run();
//				}
//				else
//				{
//					pitem.delapsed += morphFilter->run();
//					holder = oclImage2DHolder(morphFilter->outputDeviceImage());
//				}
//			}
//		}

//		if(!glInterop)
//		{
//			// Jesli nie dzielimy zasobow, trzeba je teraz sciagnac
//			pitem.dst = c.readImageFromDevice(holder);
//			qDebug("Transfering output image from the device took %.05lf ms\n",
//				c.oclElapsedEvent(holder.evt));
//			pitem.delapsed += c.oclElapsedEvent(holder.evt);
//		}
//		else
//		{
//			shareWidget->doneCurrent();
//		}

//		emit processingDone(pitem);
//	}

//	// cleanup
//	delete bayerFilter;
//	delete morphFilter;
//	delete hitmissFilter;
}
