#include "ocloclProcThread.h"
#include "cvutils.h"

#include <QDebug>

oclProcThread::oclProcThread(BlockingQueue<ProcessingItem>& queue)
	: QThread(nullptr)
	, queue(queue)
	, stopped(false)
	
	, bayerFilter(&c)
	, morphFilter(&c)
	, hitmissFilter(&c)
{
}

oclProcThread::~oclProcThread()
{
}

void oclProcThread::stop()
{
	QMutexLocker locker(&stopMutex);
	stopped = true;
}

void oclProcThread::run()
{
	while(true)
	{
		ProcessingItem item = queue.dequeue();

		{
			QMutexLocker locker(&stopMutex);
			if(stopped)
				break;
		}

		//qDebug() << endl << "New processing job:" << "\n\toperation:" << 
		//	item.op << "\n\tbayer code:" << item.bc << 
		//	"\n\tnegate:" << item.negate << endl;

		// Brak operacji, zwroc obraz zrodlowy
		if(!item.negate && 
			item.bc == cvu::BC_None &&
			item.op == cvu::MO_None)
		{
			ProcessedItem pitem = {
				/*.iters = */ 0,
				/*.delapsed = */ 0.0,
				/*.dst = */ item.src
			};
			emit processingDone(pitem);
			continue;
		}

		ProcessedItem pitem = {
			/*.iters = */ 1,
			/*.delapsed = */ 0.0,
			// Tworzymy obraz wynikowy (alokacja miejsca)
			/*.dst = */ cv::Mat(item.src.size(), 
				CV_MAKETYPE(item.src.depth(), item.src.channels()))
		};

		// TODO liczymy czas tego?
		// Zaneguj obraz
		if(item.negate)
		{
			cvu::negate(item.src, pitem.dst);
			item.src = pitem.dst;
		}
		
		// Co gdy tylko negujemy?
		
		oclImage2DHolder holder = c.copyImageToDevice(item.src, ReadOnly);
		// Tego tez mozemy liczyc czas
		qDebug("Transfering source image to the device took %.05lf ms\n", 
			c.oclElapsedEvent(holder.evt));
			
		// Filtr Bayer'a
		if(item.bc != cvu::BC_None)
		{
			//cvu::bayerFilter(item.src, pitem.dst, item.bc);
			//item.src = pitem.dst;
			
			bayerFilter.setBayerFilter(item.bc);
			bayerFilter.setSourceImage(holder);
			pitem.delapsed += bayerFilter.run();
			holder = oclImage2DHolder(bayerFilter.outputDeviceImage());
		}

		// Operacja morfologiczna/hitmiss
		if(item.op != cvu::MO_None)
		{
			//pitem.iters = cvu::morphEx(item.src, pitem.dst, item.op, item.se);
			
			if (item.op == cvu::MO_Outline ||
				item.op == cvu::MO_Skeleton ||
				item.op == cvu::MO_Skeleton_ZhangSuen)
			{
				hitmissFilter.setHitMissOperation(item.op);
				hitmissFilter.setSourceImage(holder);		
				pitem.delapsed += hitmissFilter.run();
				holder = oclImage2DHolder(morphFilter.outputDeviceImage());
			}
			else
			{
				morphFilter.setMorphologyOperation(item.op);
				morphFilter.setStructuringElement(item.se);
				morphFilter.setSourceImage(holder);
				pitem.delapsed += morphFilter.run();
				holder = oclImage2DHolder(morphFilter.outputDeviceImage());
			}
		}
		
		pitem.dst = c.readImageFromDevice(holder);
		qDebug("Transfering output image from the device took %.05lf ms\n", 
			c.oclElapsedEvent(dst.evt));

		emit processingDone(pitem);
	}
}