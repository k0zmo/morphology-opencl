#include "glwidget.h"

#include "oclthread.h"
#include "cvutils.h"
#include "elapsedtimer.h"

#include <QDebug>

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

	//if(shareWidget)
	//	shareWidget->makeCurrent();

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
		// Moze pozniej to zmienie (bedzie oczekiwac tych wartosci w argumentach funkcji)
		shareWidget->makeCurrent();

		//qDebug() << "initContext" << wglGetCurrentDC() << wglGetCurrentContext();

		c.createContextGL(platformId);
		shareWidget->doneCurrent();
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

	QString erode = "erode_c4";
	QString dilate = "dilate_c4";
	QString gradient = "gradient_c4";

	bayerFilter = new oclBayerFilter(&c);
	morphFilter = new oclMorphFilter(&c,
		conf.erode_2d.toStdString().c_str(),
		conf.dilate_2d.toStdString().c_str(),
		conf.gradient_2d.toStdString().c_str());
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
				/*.glsize = */ cv::Size(0, 0)
			};
			emit processingDone(pitem);
			continue;
		}

		ProcessedItem pitem = {
			/*.iters = */ 1,
			/*.delapsed = */ 0.0,
			// Tworzymy obraz wynikowy (alokacja miejsca)
			/*.dst = */ cv::Mat(item.src.size(),
				CV_MAKETYPE(item.src.depth(), item.src.channels())),
			/*glsize = */ cv::Size(0, 0)
		};

		//ProcessedItem pitem = {
		//	/*.iters = */ 1,
		//	/*.delapsed = */ 0.0,
		//	/*.dst = */ cv::Mat(),
		//	/*glsize = */ item.src.size()
		//};

		//shareWidget->makeCurrent();
		//glBindTexture(GL_TEXTURE_2D, 1);
		//GLint p;
		//glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &p);
		//qDebug() << p;
		//shareWidget->doneCurrent();

		//glWidget->makeCurrent();
		//((DummyGLWidget*)glWidget)->resizeSurface(
		//	pitem.glsize.width, pitem.glsize.height);

		// Zaneguj obraz
		if(item.negate)
		{
			ElapsedTimer t;
			t.start();

			cv::Mat tmp(item.src.size(), item.src.depth(), item.src.channels());
			cvu::negate(item.src, tmp);
			item.src = tmp;

			pitem.delapsed += t.elapsed();
		}
		
		oclImage2DHolder holder = c.copyImageToDevice(item.src, ReadOnly);

		// Tego tez mozemy liczyc czas
		qDebug("Transfering source image to the device took %.05lf ms\n", 
			c.oclElapsedEvent(holder.evt));
		pitem.delapsed += c.oclElapsedEvent(holder.evt);
			
		// Filtr Bayer'a
		if(item.bc != cvu::BC_None)
		{

			bayerFilter->setBayerFilter(item.bc);
			bayerFilter->setSourceImage(holder);
			pitem.delapsed += bayerFilter->run();
			holder = oclImage2DHolder(bayerFilter->outputDeviceImage());
		}

		// Operacja morfologiczna/hitmiss
		if(item.op != cvu::MO_None)
		{
			if (item.op == cvu::MO_Outline ||
				item.op == cvu::MO_Skeleton ||
				item.op == cvu::MO_Skeleton_ZhangSuen)
			{
				hitmissFilter->setHitMissOperation(item.op);
				hitmissFilter->setSourceImage(holder);
				pitem.delapsed += hitmissFilter->run();
				holder = oclImage2DHolder(morphFilter->outputDeviceImage());
			}
			else
			{
				morphFilter->setMorphologyOperation(item.op);
				morphFilter->setStructuringElement(item.se);
				morphFilter->setSourceImage(holder);

				//oclImage2DHolder output = c.createDeviceImageGL(surface, WriteOnly);
				//morphFilter->setOutputDeviceImage(output);

				pitem.delapsed += morphFilter->run();
				holder = oclImage2DHolder(morphFilter->outputDeviceImage());
			}
		}
		
		pitem.dst = c.readImageFromDevice(holder);
		qDebug("Transfering output image from the device took %.05lf ms\n",
			c.oclElapsedEvent(holder.evt));

		//pitem.delapsed += c.oclElapsedEvent(holder.evt);

		emit processingDone(pitem);
	}

	// cleanup
	delete bayerFilter;
	delete morphFilter;
	delete hitmissFilter;

	//if(glWidget) glWidget->deleteLater();
}
