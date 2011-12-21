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
EOperationType operationType(const QString& str)
{
	if(str == "erode")
		return OT_Erode;
	else if(str == "dilate")
		return OT_Dilate;
	else if(str == "open")
		return OT_Open;
	else if(str == "close")
		return OT_Close;
	else if(str == "gradient")
		return OT_Gradient;
	else if(str == "tophat")
		return OT_TopHat;
	else if(str == "blackhat")
		return OT_BlackHat;
	else if(str == "outline")
		return OT_Outline;
	else if(str == "skeleton")
		return OT_Skeleton;
	else if(str == "skeletonZH")
		return OT_Skeleton_ZhangSuen;
	return OT_Erode;
}
// -------------------------------------------------------------------------
EStructuringElementType structuringElement(const QString& str)
{
	if(str == "rect")
		return SET_Rect;
	else if(str == "ellipse")
		return SET_Ellipse;
	else if(str == "cross")
		return SET_Cross;
	else if(str == "diamond")
		return SET_Diamond;
	return SET_Rect;
}
// -------------------------------------------------------------------------
int main(int argc, char *argv[])
{
	QCoreApplication app(argc, argv);
	QTextStream qout(stdout);

	QSettings settings("./settings.cfg", QSettings::IniFormat);
	QString filename = settings.value("gui/defaultimage", "lena.jpg").toString();
	cv::Mat src = cv::imread(filename.toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat dst;

	QFile data("output.txt");
	if(!data.open(QFile::WriteOnly | QFile::Truncate))
		exit(1);
	QTextStream fout(&data);

	// ----- Ustawienia testow ----- //
	int radiusmin = settings.value("test/radiusmin", 1).toInt();
	int radiusmax = settings.value("test/radiusmax", 1).toInt();
	bool recompile = settings.value("test/recompile", false).toBool();
	int nitersopencv = settings.value("test/nitersopencv", 5).toInt();
	int	nitersopencl = settings.value("test/nitersopencl", 10).toInt();
	EStructuringElementType set = structuringElement(
		settings.value("test/structuringelement", "ellipse").toString());
	EOperationType opType = operationType(
		settings.value("test/operation", "erode").toString());

	qout << "**************************\nTest parameters: \n" 
		 << "Operation: " << settings.value("test/operation", "erode").toString() << endl
		 << "Structuring element: " << settings.value("test/structuringelement", "ellipse").toString() << endl 
		 << "Radius range: " << radiusmin << "-" << radiusmax << endl;

	// ----- Test OpenCV ----- //
	if(argc == 2 && !strcmp(argv[1], "opencv"))
	{
		qout << "Number of iterations: " << nitersopencv << endl
			 << "**************************" << endl;

		for(int radius = radiusmin; radius <= radiusmax; ++radius)
		{
			qout << "\nSize: " << 2*radius+1 << "x" << 2*radius+1 << ":\n";
			qout.flush();

			cv::Mat element = standardStructuringElement(radius, radius, set);

			for(int i = 0; i < nitersopencv; ++i)
			{
				LARGE_INTEGER freq, start, end;
				QueryPerformanceFrequency(&freq);
				QueryPerformanceCounter(&start);

				if (opType == OT_Outline ||
					opType == OT_Skeleton ||
					opType == OT_Skeleton_ZhangSuen)
				{
					switch (opType)
					{
					case OT_Outline:
							morphologyOutline(src, dst);
							break;
					case OT_Skeleton:
							morphologySkeleton(src, dst);
							break;
					case OT_Skeleton_ZhangSuen:
							morphologySkeletonZhangSuen(src, dst);
							break;
					default: break;
					}
				}
				else
				{
					int op_type;
					switch(opType)
					{
					case OT_Erode: op_type = cv::MORPH_ERODE; break;
					case OT_Dilate: op_type = cv::MORPH_DILATE; break;
					case OT_Open: op_type = cv::MORPH_OPEN; break;
					case OT_Close: op_type = cv::MORPH_CLOSE; break;
					case OT_Gradient: op_type = cv::MORPH_GRADIENT; break;
					case OT_TopHat: op_type = cv::MORPH_TOPHAT; break;
					case OT_BlackHat: op_type = cv::MORPH_BLACKHAT; break;
					default: op_type = cv::MORPH_ERODE; break;
					}

					cv::morphologyEx(src, dst, op_type, element);
				}

				QueryPerformanceCounter(&end);
				double elapsed = (static_cast<double>(end.QuadPart - start.QuadPart) / 
					static_cast<double>(freq.QuadPart)) * 1000.0f;

				fout << elapsed << endl;
				qout << elapsed << endl;
			}
		}
	}
	// ----- Test OpenCL ----- //
	else
	{
		qout << "Number of iterations: " << nitersopencl << endl
			 << "Recompile for each SE: " << recompile << endl
			 << "**************************" << endl;

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
		ocl->setSourceImage(&src);

		for(int radius = radiusmin; radius <= radiusmax; ++radius)
		{
			cv::Mat element = standardStructuringElement(radius, radius, set);
			qout << "\nSize: " << 2*radius+1 << "x" << 2*radius+1 << ":\n";
			//fout << "\nSize: " << 2*radius+1 << "x" << 2*radius+1 << ":\n";
			qout.flush();

			int coords_size = ocl->setStructuringElement(element);

			if(recompile)
				ocl->recompile(opType, coords_size);

			for(int i = 0; i < nitersopencl; ++i)
			{
				int iters;

				// "Rozgrzej karte" - pierwsze 2-3 iteracje beda wolniejsze
				double delapsed = ocl->morphology(opType, dst, iters);

				// Zapisz statystyki
				fout << delapsed << endl;
			}
		}
	}

	// Pokaz/Zapisz obraz wynikowy
	cv::imshow("Test", dst);
	cv::imwrite("output.png", dst);

	return app.exec();
}
