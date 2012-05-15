#include "glwidget.h"

#include <qclcontext.h>
#include <qclcontextgl.h>

#include "oclfilter.h"
#include "oclbayerfilter.h"
#include "oclmorphfilter.h"
#include "oclutils.h"
#include "oclmorphhitmissfilter.h"

#include "oclthread.h"
#include "cvutils.h"
#include "elapsedtimer.h"

#include <QDebug>

#ifdef Q_WS_X11
#	include <GL/glx.h>
#endif

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

void oclThread::setSharedWidget(GLDummyWidget* shareWidget)
{
	if(!isRunning())
		this->shareWidget = shareWidget;
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

void oclThread::initContext(QCLContext* ctx)
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
	ctx->create(devList);

	QCLCommandQueue cq = ctx->createCommandQueue(CL_QUEUE_PROFILING_ENABLE);
	ctx->setCommandQueue(cq);
}

void oclThread::initContextWithGL(QCLContextGL* ctx)
{
	// TODO: reuse from queryPlatforms
	QList<QCLPlatform> pls = QCLPlatform::platforms();

	if(platformId >= pls.count())
	{
		success = false;
		return;
	}

	// TODO: Trzeba zmodyfikowac zrodla QCLContextGL
	QCLPlatform pl = pls[platformId];
	//QList<QCLDevice> devs = QCLDevice::devices(QCLDevice::GPU, pl);

	//if(deviceId >= devs.count())
	//{
	//	success = false;
	//	return;
	//}
	
	//QCLDevice dev = devs[deviceId];
	//QList<QCLDevice> devList;
	//devList << dev;
	//ctx->create(devList);

	Q_ASSERT(shareWidget);

	shareWidget->makeCurrent();
	ctx->create(pl);

	qDebug() << ctx->defaultDevice().name();

	QCLCommandQueue cq = ctx->createCommandQueue(CL_QUEUE_PROFILING_ENABLE);
	ctx->setCommandQueue(cq);
}

void oclThread::choose(int platformId_, int deviceId_)
{
	platformId = platformId_;
	deviceId = deviceId_;
}

void oclThread::run()
{
	QCLContext* ctx;
	QCLContextGL* ctxg = nullptr;
	if(shareWidget)
	{
		ctx = new QCLContextGL;
		ctxg = static_cast<QCLContextGL*>(ctx);
		initContextWithGL(ctxg);
	}
	else
	{
		ctx = new QCLContext;
		initContext(ctx);
	}

	cl_int err = ctx->lastError();
	if(!success || err != CL_SUCCESS)
	{
		emit openCLInitialized(false);
		return;
	}

	// Blok filtracji bayera
	oclBayerFilter bayerFilter(ctx);
	success = ctx->lastError() == CL_SUCCESS;
	if(!success)
	{
		emit openCLInitialized(false);
		return;
	}

	// Blok filtracji morfologicznej
	oclMorphFilter morphFilter
		(ctx, conf.erode_2d.toAscii().constData(),
		 conf.dilate_2d.toAscii().constData(), 
		 conf.gradient_2d.toAscii().constData());
	if(!success)
	{
		emit openCLInitialized(false);
		return;
	}

	// Blok filtracji morfologicznej typu Hit-Miss
	oclMorphHitMissFilter hitmissFilter
		(ctx, conf.atomicCounters);
	if(!success)
	{
		emit openCLInitialized(false);
		return;
	}

	// Udalo nam sie ruszyc z OpenCLem
	emit openCLInitialized(true);

	bool glInterop = shareWidget != nullptr;
	if(ctxg) glInterop &= ctxg->supportsObjectSharing();
	else glInterop = false;

	printf("GL/CL Interop: %s\n", glInterop
		? "yes" : "no");

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
		#else
		qDebug();
		#endif

		ProcessedItem pitem = {
			/*.iters = */ 1,
			/*.delapsed = */ 0.0,
			/*.dst = */ cv::Mat(),
			/*.glsize = */ cv::Size(0, 0)// <- bez interopu
		};

		// Brak operacji, zwroc obraz zrodlowy
		if(!item.negate &&
			item.bc == cvu::BC_None &&
			item.op == cvu::MO_None)
		{
			pitem.iters = 0;
			pitem.dst = item.src;
			emit processingDone(pitem);
			continue;
		}

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

		if(glInterop) 
		{
			pitem.glsize = item.src.size();
			shareWidget->resizeSurface(pitem.glsize.width, pitem.glsize.height);
		}

		// Utworz obraz na urzadzeniu
		QCLImage2D holder = ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), 
				 QSize(item.src.cols, item.src.rows),
				 QCLMemoryObject::ReadOnly);
		
		// Skopiuj do niego dane
		QCLEvent evt = holder.writeAsync
				(const_cast<uchar*>(item.src.ptr<uchar>()),
				 QRect(0, 0, item.src.cols, item.src.rows),
				 QCLEventList(), 0);
		evt.waitForFinished();

		// Tego tez mozemy liczyc czas
		qreal elapsed = oclUtils::eventDuration(evt);
		qDebug("\nTransfering source image to the device took %.05lf ms", elapsed);
		pitem.delapsed += elapsed;
		
		// Filtr Bayer'a
		if(item.bc != cvu::BC_None)
		{
			bayerFilter.setBayerFilter(item.bc);
			bayerFilter.setSourceImage(holder);

			// Czy filtr bayera jest ostatnim 'bloczkiem'
			if(glInterop && item.op == cvu::MO_None)
			{
				QCLImage2D output = ctxg->createTexture2D
					(shareWidget->surface(), QCLMemoryObject::WriteOnly);
				bayerFilter.setOutputDeviceImage(output);
				pitem.delapsed += bayerFilter.run();
			}
			else
			{
				pitem.delapsed += bayerFilter.run();
				holder = bayerFilter.outputDeviceImage();
			}
		}

		// Operacja morfologiczna/hitmiss
		if(item.op != cvu::MO_None)
		{	
			if(isHitMiss(item.op))
			{
				hitmissFilter.setHitMissOperation(item.op);
				hitmissFilter.setSourceImage(holder);

				pitem.delapsed += hitmissFilter.run();
				holder = hitmissFilter.outputDeviceImage();

				if(glInterop)
				{
					QCLImage2D output = ctxg->createTexture2D
						(shareWidget->surface(), QCLMemoryObject::WriteOnly);
					hitmissFilter.setOutputDeviceImage(output);
					pitem.delapsed += hitmissFilter.run();
				}
				else
				{
					pitem.delapsed += hitmissFilter.run();
					holder = hitmissFilter.outputDeviceImage();
				}
			}
			else
			{
				morphFilter.setMorphologyOperation(item.op);
				morphFilter.setStructuringElement(item.se);
				morphFilter.setSourceImage(holder);

				if(glInterop)
				{
					QCLImage2D output = ctxg->createTexture2D
						(shareWidget->surface(), QCLMemoryObject::WriteOnly);
					morphFilter.setOutputDeviceImage(output);
					pitem.delapsed += morphFilter.run();
				}
				else
				{
					pitem.delapsed += morphFilter.run();
					holder = morphFilter.outputDeviceImage();
				}
			}
		}

		if(!glInterop)
		{
			// Jesli nie dzielimy zasobow, trzeba je teraz sciagnac
			int format = CV_8U;
			pitem.dst = cv::Mat
				(cv::Size(holder.width(), holder.height()),
				format, cv::Scalar(1));

			evt = holder.readAsync
				(pitem.dst.ptr<uchar>()/* + 1 + pitem.dst.cols*/,
				//QRect(1, 1, pitem.dst.cols-2, pitem.dst.rows-2),
				QRect(0, 0, pitem.dst.cols, pitem.dst.rows),
				QCLEventList(), pitem.dst.cols);
			evt.waitForFinished();

			elapsed = oclUtils::eventDuration(evt);
			qDebug("Transfering output image from the device took %.05lf ms", elapsed);
			pitem.delapsed += elapsed;
		}
		
		emit processingDone(pitem);
	}

	delete ctx;
}
