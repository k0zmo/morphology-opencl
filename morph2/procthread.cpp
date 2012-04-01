#include "procthread.h"
#include "elapsedtimer.h"
#include "cvutils.h"

#include <QDebug>

ProcThread::ProcThread(BlockingQueue<ProcessingItem>& queue)
	: QThread(nullptr)
	, queue(queue)
	, stopped(false)
{
}

ProcThread::~ProcThread()
{
}

void ProcThread::stop()
{
	QMutexLocker locker(&stopMutex);
	stopped = true;
}

void ProcThread::run()
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

		// Zacznij liczyc czas
		ElapsedTimer timer;
		timer.start();

		ProcessedItem pitem = {
			/*.iters = */ 1,
			/*.delapsed = */ 0.0,
			// Tworzymy obraz wynikowy (alokacja miejsca)
			/*.dst = */ cv::Mat(item.src.size(), 
				CV_MAKETYPE(item.src.depth(), item.src.channels()))
		};

		// Zaneguj obraz
		if(item.negate)
		{
			cvu::negate(item.src, pitem.dst);
			item.src = pitem.dst;
		}

		// Filtr Bayer'a
		if(item.bc != cvu::BC_None)
		{
			cvu::bayerFilter(item.src, pitem.dst, item.bc);
			item.src = pitem.dst;
		}

		// Operacja morfologiczna
		if(item.op != cvu::MO_None)
		{
			pitem.iters = cvu::morphEx(item.src, pitem.dst, item.op, item.se);
		}

		pitem.delapsed = timer.elapsed();
		emit processingDone(pitem);
	}
}