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

#include "morphocl.h"
#include "morphop.h"

// -------------------------------------------------------------------------
int main(int argc, char *argv[])
{
	QCoreApplication app(argc, argv);
	QTextStream qout(stdout);

	QSettings settings("./settings.cfg", QSettings::IniFormat);

	int method = settings.value("opencl/method", 0).toInt();
	int device = settings.value("opencl/device", 0).toInt();

	MorphOpenCL* ocl;
	if(method == 0) ocl = new MorphOpenCLImage();
	else ocl = new MorphOpenCLBuffer();

	ocl->errorCallback = [&qout](const QString& message, cl_int err)
	{
		Q_UNUSED(err);
		qout << "OpenCL error: " <<  message << " Error code: " << 
			MorphOpenCL::openCLErrorCodeStr(err) << endl;
		exit(1);
	};

	cl_device_type deviceType = CL_DEVICE_TYPE_CPU;
	if(device == 1) deviceType = CL_DEVICE_TYPE_GPU;
	QString dev = (device == 1) ? ("GPU") : ("CPU");

	qout << "Creating OpenCL Stuff [" << dev << "]..." << endl;
	
	if(!ocl->initOpenCL(deviceType))
	{
		qout << "Opencl init failed" << endl;
		exit(1);
	}

	qout << "Loading image..." << endl;
	QString filename = "bin1.png";
	cv::Mat src = cv::imread(filename.toStdString(), CV_LOAD_IMAGE_GRAYSCALE);

	ocl->setSourceImage(&src);

	qout << "DONE" << endl;

	QFile data("output.txt");
	if(!data.open(QFile::WriteOnly | QFile::Truncate))
		exit(1);
	QTextStream fout(&data);

	fout << "method: " << settings.value("opencl/method", 0).toInt() << endl;
	fout << "workgroupsizex: " << settings.value("misc/workgroupsizex", 0).toInt() << endl;
	fout << "workgroupsizey: " << settings.value("misc/workgroupsizey", 0).toInt() << endl;
	fout << "readingmethod: " << settings.value("misc/readingmethod", 0).toInt() << endl;
	fout << "kernel: " << settings.value("kernel/dilate", "").toString() << endl;
	
	cv::Mat dst, element = standardStructuringElement(radius, radius, SET_Ellipse);
	EOperationType opType = OT_Thinning;
	/*int coords_size =*/ ocl->setStructureElement(element);

	//for(int radius = 1; radius <= 35; ++radius)
	{
		int radius = 7;
		qout << "\nSize: " << 2*radius+1 << "x" << 2*radius+1 << ":\n";
		fout << "\nSize: " << 2*radius+1 << "x" << 2*radius+1 << ":\n";

		for(int i = 0; i < 10; ++i)
		{
			int iters;

			// "Rozgrzej karte" - pierwsze 2-3 iteracje beda wolniejsze
			double delapsed = ocl->morphology(opType, dst, iters);

			// Wyswietl statystyki
			qout << "Time elasped : " << delapsed << " ms, iterations: " << iters << endl;
			fout << "Time elasped : " << delapsed << " ms, iterations: " << iters << endl;
		}
	}
	
	// Pokaz/Zapisz obraz wynikowy
	cv::imshow("Test", dst);
	//cv::imwrite("output.png", dst);

	return app.exec();
}
