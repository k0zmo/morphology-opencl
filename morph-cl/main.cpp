#include <QtCore/QCoreApplication>
#include <QTextStream>
#include <QSettings>
#include <QFile>

#define CV_NO_BACKWARD_COMPATIBILITY

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <CL/cl.hpp>

#if !defined(_WIN32)
#include <sys/time.h>
#endif

#include "morphoclimage.h"
#include "morphoclbuffer.h"
#include "morphop.h"

// -------------------------------------------------------------------------
int main(int argc, char *argv[])
{
	QCoreApplication app(argc, argv);
	QTextStream qout(stdout);

	int method;
	printf("There are 2 methods implemented:\n"
		"\t1) Images\n"
		"\t2) Buffers\n"
		"Choose method: ");
	scanf("%d", &method);

	MorphOpenCL* ocl;
	if(method == 1) ocl = new MorphOpenCLImage();
	else ocl = new MorphOpenCLBuffer();

	ocl->errorCallback = [&qout](const QString& message, cl_int err)
	{
		Q_UNUSED(err);
		qout << "OpenCL error: " <<  message << " Error code: " << 
			MorphOpenCL::openCLErrorCodeStr(err) << endl;
		exit(1);
	};
	
	if(!ocl->initOpenCL())
	{
		qout << "Opencl init failed" << endl;
		exit(1);
	}

	QSettings settings("./settings.cfg", QSettings::IniFormat);
	qout << "Loading image..." << endl;
	QString filename = settings.value("gui/defaultimage", "lena.jpg").toString();
	cv::Mat src = cv::imread(filename.toStdString(), CV_LOAD_IMAGE_GRAYSCALE);

	ocl->setSourceImage(&src);

	qout << "DONE" << endl;

	QFile data("output.txt");
	if(!data.open(QFile::WriteOnly | QFile::Truncate))
		exit(1);
	QTextStream fout(&data);

	fout << "method: " << settings.value("opencl/method", 0).toInt() << endl;
	fout << "workgroupsizex: " << settings.value("opencl/workgroupsizex", 0).toInt() << endl;
	fout << "workgroupsizey: " << settings.value("opencl/workgroupsizey", 0).toInt() << endl;
	fout << "kernel: " << settings.value("kernel/erode", "").toString() << endl;

	EOperationType opType = OT_Gradient;
	cv::Mat dst;

	for(int radius = 1; radius <= 35; ++radius)
	{
		//int radius = 1;
		cv::Mat element = standardStructuringElement(radius, radius, SET_Ellipse);
#if 1
		int coords_size = ocl->setStructureElement(element);
		//ocl->recompile(OT_Gradient, coords_size);
		
		qout << "\nSize: " << 2*radius+1 << "x" << 2*radius+1 << ":\n";
		fout << "\nSize: " << 2*radius+1 << "x" << 2*radius+1 << ":\n";
		qout.flush();

		for(int i = 0; i < 12; ++i)
		{
			int iters;

			// "Rozgrzej karte" - pierwsze 2-3 iteracje beda wolniejsze
			double delapsed = ocl->morphology(opType, dst, iters);

			// Zapisz statystyki
			fout << "Time elasped : " << delapsed << " ms, iterations: " << iters << endl;
		}
#else
		qout << "\nSize: " << 2*radius+1 << "x" << 2*radius+1 << ":\n";
		qout.flush();

		for(int i = 0; i < 5; ++i)
		{
			LARGE_INTEGER freq, start, end;
			QueryPerformanceFrequency(&freq);
			QueryPerformanceCounter(&start);

			//morphologyOutline(src, dst);
			cv::morphologyEx(src, dst, cv::MORPH_GRADIENT, element);
			//morphologyErode(src, dst, element);

			QueryPerformanceCounter(&end);
			double elapsed = (static_cast<double>(end.QuadPart - start.QuadPart) / 
				static_cast<double>(freq.QuadPart)) * 1000.0f;

			fout << elapsed << endl;
			qout << elapsed << endl;
		}
#endif
	}

	// Pokaz/Zapisz obraz wynikowy
	cv::imshow("Test", dst);
	cv::imwrite("output.png", dst);

	return app.exec();
}
