#include "glwidget.h"

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
	, bayerFilter(nullptr)
	, morphFilter(nullptr)
	, hitmissFilter(nullptr)
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
	std::vector<oclPlatformDesc> pl;
	PlatformDevicesMap plToDevs;

	oclContext::cb = [this](const std::string& msg, cl_int err)
	{
		qDebug() << QString::fromStdString(oclContext::oclErrorString(err))
				 << ": " << QString::fromStdString(msg) << endl;
		success = false;
	};

	c.retrievePlatforms(pl);

	// Pobierz wszystkie urzadzenia z danej platformy
	std::for_each(pl.begin(), pl.end(),
		[&](const oclPlatformDesc& desc)
	{
		std::vector<oclDeviceDesc> devs;
		c.retrieveDevices(desc.id, devs);

		// konwersja std::vector -> QList
		QList<oclDeviceDesc> qdevs;
		std::for_each(devs.begin(), devs.end(),
			[&qdevs](const oclDeviceDesc& d)
		{
			qdevs.append(d);
		});

		plToDevs.insert(desc, qdevs);
	});

	return plToDevs;
}

void oclThread::initContext()
{
	if(shareWidget)
	{
		// createContextGL oczekuje aktywnego kontekstu
		// oclThread ma kontekst na wylacznosc dlatego
		// nie potrzebne sa zadne doneCurrent i pozniejsze makeCurrent
		// (o ile nikt nie zawola gdzies doneCurrent)
		shareWidget->makeCurrent();

#ifdef Q_WS_WIN32
		HDC dc = wglGetCurrentDC();
		HGLRC rc = wglGetCurrentContext();
#else
		Display* dc = glXGetCurrentDisplay();
		GLXContext rc = glXGetCurrentContext();
#endif

		if(!dc || !rc)
		{
			c.createContext(platformId);
			shareWidget = nullptr;
		}
		else
		{
			c.createContextGL(platformId);
		}
	}
	else
	{
		c.createContext(platformId);
	}

	// wybor urzadzenia
	c.chooseDevice(deviceId);

	c.createCommandQueue(true);
	c.setWorkgroupSize(conf.workgroupSizeX,
		conf.workgroupSizeY);

	// Bloczek z filtracja bayera
	bayerFilter = new oclBayerFilter(&c);

	// Bloczek z operacjami morfologicznymi
	morphFilter = new oclMorphFilter(&c,
		conf.erode_2d.toStdString().c_str(),
		conf.dilate_2d.toStdString().c_str(),
		conf.gradient_2d.toStdString().c_str());

	// Bloczek z operacjami morfologicznymi typu hit-miss
	hitmissFilter = new oclMorphHitMissFilter(&c,
		conf.atomicCounters);
}

void oclThread::choose(int platformId_, int deviceId_)
{
	platformId = platformId_;
	deviceId = deviceId_;
}

void oclThread::run()
{
	initContext();
	emit openCLInitialized(success);
	if(!success)
		return;

	bool glInterop = shareWidget != nullptr;

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

		if(glInterop) 
		{
			pitem.glsize = item.src.size();
			shareWidget->resizeSurface(pitem.glsize.width, pitem.glsize.height);
		}
		
		oclImage2DHolder holder = c.copyImageToDevice(item.src, ReadOnly);

		// Tego tez mozemy liczyc czas
		qDebug("Transfering source image to the device took %.05lf ms", 
			c.oclElapsedEvent(holder.evt));
		pitem.delapsed += c.oclElapsedEvent(holder.evt);
			
		// Filtr Bayer'a
		if(item.bc != cvu::BC_None)
		{
			bayerFilter->setBayerFilter(item.bc);
			bayerFilter->setSourceImage(holder);

			// Czy filtr bayera jest ostatnim 'bloczkiem'
			if(glInterop && item.op == cvu::MO_None)
			{
				oclImage2DHolder output = c.createDeviceImageGL(shareWidget->surface(), WriteOnly);
				bayerFilter->setOutputDeviceImage(output);
				pitem.delapsed += bayerFilter->run();
			}
			else
			{
				pitem.delapsed += bayerFilter->run();
				holder = oclImage2DHolder(bayerFilter->outputDeviceImage());
			}
		}

		// Operacja morfologiczna/hitmiss
		if(item.op != cvu::MO_None)
		{
			if(isHitMiss(item.op))
			{
				hitmissFilter->setHitMissOperation(item.op);
				hitmissFilter->setSourceImage(holder);

				if(glInterop)
				{
					oclImage2DHolder output = c.createDeviceImageGL(shareWidget->surface(), WriteOnly);
					hitmissFilter->setOutputDeviceImage(output);
					pitem.delapsed += hitmissFilter->run();
				}
				else
				{
					pitem.delapsed += hitmissFilter->run();
					holder = oclImage2DHolder(morphFilter->outputDeviceImage());
				}
			}
			else
			{
				morphFilter->setMorphologyOperation(item.op);
				morphFilter->setStructuringElement(item.se);
				morphFilter->setSourceImage(holder);

				if(glInterop)
				{
					oclImage2DHolder output = c.createDeviceImageGL(shareWidget->surface(), WriteOnly);
					morphFilter->setOutputDeviceImage(output);
					pitem.delapsed += morphFilter->run();
				}
				else
				{
					pitem.delapsed += morphFilter->run();
					holder = oclImage2DHolder(morphFilter->outputDeviceImage());
				}
			}
		}

		if(!glInterop)
		{
			// Jesli nie dzielimy zasobow, trzeba je teraz sciagnac
			pitem.dst = c.readImageFromDevice(holder);
			qDebug("Transfering output image from the device took %.05lf ms",
				c.oclElapsedEvent(holder.evt));
			pitem.delapsed += c.oclElapsedEvent(holder.evt);
		}

		emit processingDone(pitem);
	}

	// cleanup
	delete bayerFilter;
	delete morphFilter;
	delete hitmissFilter;
}
