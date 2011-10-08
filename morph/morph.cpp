#include "morph.h"
#include "morphOp.h"

#include <QElapsedTimer>
#include <QFileDialog>
#include <QMessageBox>
#include <QTextStream>

#if !defined(_WIN32)
#include <sys/time.h>
#endif

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// Morph

Morph::Morph(QString filename, QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	// Menu
	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(openTriggered()));
	connect(ui.actionSave, SIGNAL(triggered()), this, SLOT(saveTriggered()));
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(exitTriggered()));
	connect(ui.actionOpenCL, SIGNAL(triggered(bool)), this, SLOT(openCLTriggered(bool)));

	connect(ui.cbInvert, SIGNAL(stateChanged(int)), this, SLOT(invertChanged(int)));

	// Operacje
	connect(ui.rbNone, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbErode, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbDilate, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbOpen, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbClose, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbGradient, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbTopHat, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbBlackHat, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbRemove, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbSkeleton, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));
	connect(ui.rbVoronoi, SIGNAL(toggled(bool)), this, SLOT(operationToggled(bool)));

	//connect(ui.sbPruning, SIGNAL(valueChanged(int)), this, SLOT(pruningItersChanged(int)));
	connect(ui.sbPruning, SIGNAL(valueChanged(int)), this, SLOT(pruneChanged(int)));
	connect(ui.cbPrune, SIGNAL(stateChanged(int)), this, SLOT(pruneChanged(int)));

	// Element strukturalny
	connect(ui.rbRect, SIGNAL(toggled(bool)), this, SLOT(structureElementToggled(bool)));
	connect(ui.rbEllipse, SIGNAL(toggled(bool)), this, SLOT(structureElementToggled(bool)));
	connect(ui.rbCross, SIGNAL(toggled(bool)), this, SLOT(structureElementToggled(bool)));
	connect(ui.rbDiamond, SIGNAL(toggled(bool)), this, SLOT(structureElementToggled(bool)));

	// Rozmiar elementu strukturalnego
	connect(ui.cbSquare, SIGNAL(stateChanged(int)), this, SLOT(ratioChanged(int)));
	connect(ui.hsXElementSize, SIGNAL(valueChanged(int)), this, SLOT(elementSizeXChanged(int)));
	connect(ui.hsYElementSize, SIGNAL(valueChanged(int)), this, SLOT(elementSizeYChanged(int)));
	connect(ui.dialRotation, SIGNAL(valueChanged(int)), this, SLOT(rotationChanged(int)));
	connect(ui.pbResetRotation, SIGNAL(pressed()), this, SLOT(rotationResetPressed()));

	initOpenCL();
	openFile(filename);

	// Wartosci domyslne
	ui.rbNone->toggle();
	ui.rbRect->toggle();
	ui.cbSquare->setChecked(true);

	ui.lbXElementSize->setText(QString::fromLatin1("Horizontal: 3"));
	ui.lbYElementSize->setText(QString::fromLatin1("Vertical: 3"));
	ui.lbRotation->setText(QString::fromLatin1("0"));

	statusBarLabel = new QLabel();
	ui.statusBar->addPermanentWidget(statusBarLabel);

	// TODO
	ui.rbVoronoi->setVisible(false);
	ui.sbPruning->setVisible(false);
	ui.cbPrune->setVisible(false);
}
// -------------------------------------------------------------------------
Morph::~Morph()
{

}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// Zdarzenia

void Morph::openTriggered()
{
	QString filename = QFileDialog::getOpenFileName(
		nullptr, QString(), ".",
		QString::fromLatin1("Image files (*.png *.jpg *.bmp)"));

	if(!filename.isEmpty())
	{
		openFile(filename);
		refresh();
	}
}
// -------------------------------------------------------------------------
void Morph::saveTriggered()
{
	QString filename = QFileDialog::getSaveFileName(this, QString(), ".",
		QString::fromLatin1("Image file (*.png)"));
	if(!filename.isEmpty())
	{
		ui.lbImage->pixmap()->toImage().save(filename);
	}
}
// -------------------------------------------------------------------------
void Morph::exitTriggered()
{
	close();
}
// -------------------------------------------------------------------------
void Morph::openCLTriggered(bool state)
{
	Q_UNUSED(state);
	refresh();
}
// -------------------------------------------------------------------------
void Morph::invertChanged(int state)
{
	Q_UNUSED(state);
	cv::Mat lut(1, 256, CV_8U);
	uchar* p = lut.ptr<uchar>();
	for(int i = 0; i < lut.cols; ++i)
	{
		*p++ = 255 - i;
	}
	cv::LUT(src, lut, src);

	if(ui.actionOpenCL->isEnabled())
	{
		cl_int err = cq.enqueueWriteBuffer(clSrc, CL_TRUE, 0,
			src.rows * src.cols, src.ptr<uchar>());
		clError("Error while writing new data to OpenCL buffer!", err);
	}

	refresh();
}
// -------------------------------------------------------------------------
void Morph::operationToggled(bool checked)
{
	// Warunek musi byc spelniony bo sa zglaszane 2 zdarzenia
	// 1 - jeden z radiobuttonow zmienil stan z aktywnego na nieaktywny
	// 2 - zaznaczony radiobutton zmienil stan z nieaktywnego na aktywny
	if(checked)
	{
		refresh();
	}
}
// -------------------------------------------------------------------------
void Morph::structureElementToggled(bool checked)
{
	if(checked)
	{
		if(!ui.rbNone->isChecked())
			refresh();
	}
}
// -------------------------------------------------------------------------
void Morph::ratioChanged(int state)
{
	if(state == Qt::Checked)
	{
		int vv = qMax(ui.hsXElementSize->value(), ui.hsYElementSize->value());
		ui.hsXElementSize->setValue(vv);
		ui.hsYElementSize->setValue(vv);
	}
}
// -------------------------------------------------------------------------
void Morph::elementSizeXChanged(int value)
{
	Q_UNUSED(value);
	ui.lbXElementSize->setText(QString::fromLatin1("Horizontal: ") + 
		QString::number(2 * ui.hsXElementSize->value() + 1));

	if(ui.cbSquare->checkState() == Qt::Checked)
	{
		if(ui.hsYElementSize->value() != ui.hsXElementSize->value())
			ui.hsYElementSize->setValue(ui.hsXElementSize->value());
	}

	if(!ui.rbNone->isChecked())
		refresh();
}
// -------------------------------------------------------------------------
void Morph::elementSizeYChanged(int value)
{
	Q_UNUSED(value);
	ui.lbYElementSize->setText(QString::fromLatin1("Vertical: ") + 
		QString::number(2 * ui.hsYElementSize->value() + 1));

	if(ui.cbSquare->checkState() == Qt::Checked)
	{
		if(ui.hsXElementSize->value() != ui.hsYElementSize->value())
			ui.hsXElementSize->setValue(ui.hsYElementSize->value());
	}

	if(!ui.rbNone->isChecked())
		refresh();
}
// -------------------------------------------------------------------------
void Morph::rotationChanged(int value)
{
	Q_UNUSED(value);
	int angle = ui.dialRotation->value();
	if(angle >= 180) { angle -= 180; }
	else { angle += 180; }
	angle = 360 - angle;
	angle = angle % 360;

	ui.lbRotation->setText(QString::number(angle));

	if(!ui.rbNone->isChecked())
		refresh();
}
// -------------------------------------------------------------------------
void Morph::rotationResetPressed()
{
	ui.dialRotation->setValue(180);
}
// -------------------------------------------------------------------------
void Morph::pruneChanged(int state)
{
	Q_UNUSED(state);
	if(ui.rbVoronoi->isChecked() && ui.sbPruning->value() != 0)
	{
		refresh();
	}
}

// Koniec zdarzen
// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH

void Morph::showCvImage(const cv::Mat& image)
{
	auto toQImage = [](const cv::Mat& image)
	{
		return QImage(
			reinterpret_cast<const quint8*>(image.data),
			image.cols, image.rows, image.step, 
			QImage::Format_Indexed8);
	};

	ui.lbImage->setPixmap(QPixmap::fromImage(toQImage(image)));
}
// -------------------------------------------------------------------------
void Morph::openFile(const QString& filename)
{
	qsrc = QImage(filename);
	if(qsrc.format() != QImage::Format_RGB888)
		qsrc = qsrc.convertToFormat(QImage::Format_RGB888);

	auto toCvMat = [](const QImage& qimage) -> cv::Mat
	{
		cv::Mat mat(qimage.height(), qimage.width(), CV_8UC3,
			const_cast<uchar*>(qimage.bits()),
			qimage.bytesPerLine());

		// Konwersja do obrazu jednokanalowego
		cvtColor(mat, mat, CV_RGB2GRAY);
		return mat;
	};

	src = toCvMat(qsrc);
	if(ui.actionOpenCL->isEnabled())
	{
		cl_int err;
		clSrc = cl::Buffer(context,
			CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, 
			src.rows * src.cols, // obraz 1-kanalowy
			src.ptr<uchar>(), &err);
		clError("Error creating source OpenCL buffer", err);

		//cl::Image2D clSrcImage = cl::Image2D(context, 
		//	CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
		//	cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
		//	src.cols, src.rows, 0, src.ptr<uchar>(), &err);
		//clError("Error creating source OpenCL image2D", err);
	}

	this->resize(0, 0);
}
// -------------------------------------------------------------------------
cv::Mat Morph::standardStructuringElement()
{
	EStructureElementType type;

	if(ui.rbRect->isChecked()) type = SET_Rect;
	else if(ui.rbEllipse->isChecked()) type = SET_Ellipse;
	else if(ui.rbCross->isChecked()) type = SET_Cross;
	else type = SET_Diamond;

	return ::standardStructuringElement(
		ui.hsXElementSize->value(), 
		ui.hsYElementSize->value(),
		type,
		ui.dialRotation->value());
}
// -------------------------------------------------------------------------
void Morph::refresh()
{
	if(ui.rbNone->isChecked())
	{
		showCvImage(src);
		return;
	}

	// Operacje hit-miss
	if (ui.rbRemove->isChecked() ||
		ui.rbSkeleton->isChecked() || 
		ui.rbVoronoi->isChecked())
	{
		// deaktywuj wybor elementu strukturalnego
		ui.gbElement->setEnabled(false);
		ui.gbElementSize->setEnabled(false);
	}
	else
	{
		// deaktywuj wybor elementu strukturalnego
		ui.gbElement->setEnabled(true);
		ui.gbElementSize->setEnabled(true);
	}

	if(ui.actionOpenCL->isChecked())
		morphologyOpenCL();
	else
		morphologyOpenCV();
}
// -------------------------------------------------------------------------
void Morph::morphologyOpenCV()
{
#if defined(_WIN32)
	LARGE_INTEGER freq, start, end;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
#else
	timeval start, end;
	gettimeofday(&start, NULL);
#endif

	int niters = 1;

	// Operacje hit-miss
	if (ui.rbRemove->isChecked() ||
		ui.rbSkeleton->isChecked() || 
		ui.rbVoronoi->isChecked())
	{
		cv::Mat dst = src.clone();

		if(ui.rbRemove->isChecked())
		{
			morphologyRemove(src, dst);
		}
		else if(ui.rbSkeleton->isChecked())
		{
			cv::Mat src1 = src.clone();
			niters = morphologySkeleton(src1, dst);

			// Szkielet - bialy
			// tlo - szare (zmiana z bialego)
			// obiekt - czarny
			dst = src/2 + dst;
		}
		else if(ui.rbVoronoi->isChecked())
		{
			cv::Mat src1 = src.clone();
			niters = morphologyVoronoi(src1, dst,
				ui.cbPrune->isChecked() ? ui.sbPruning->value() : 0);

			// Strefy - szare
			// Reszta - niezmienione
			dst = dst/2 + src;		
		}

		showCvImage(dst);
	}
	else
	{
		int op_type = cv::MORPH_ERODE;
		if(ui.rbErode->isChecked()) { op_type = cv::MORPH_ERODE; }
		else if(ui.rbDilate->isChecked()) { op_type = cv::MORPH_DILATE; }
		else if(ui.rbOpen->isChecked()) { op_type = cv::MORPH_OPEN; }
		else if(ui.rbClose->isChecked()) { op_type = cv::MORPH_CLOSE; }
		else if(ui.rbGradient->isChecked()) { op_type = cv::MORPH_GRADIENT; }
		else if(ui.rbTopHat->isChecked()) { op_type = cv::MORPH_TOPHAT; }
		else if(ui.rbBlackHat->isChecked()) { op_type = cv::MORPH_BLACKHAT; }

		cv::Mat element = standardStructuringElement();
		cv::Mat dst;

		if(ui.rbErode->isChecked())
		{			
			doErode(src, dst, element);
		}
		else
		{
			cv::morphologyEx(src, dst, op_type, element);
		}

		showCvImage(dst);
	}
#if defined(_WIN32)
	QueryPerformanceCounter(&end);
	double elapsed = (static_cast<double>(end.QuadPart - start.QuadPart) / 
		static_cast<double>(freq.QuadPart)) * 1000.0f;
#else
	gettimeofday(&end, NULL);
	double elapsed = (static_cast<double>(end.tv_sec - start.tv_sec) * 1000 +
		0.001f * static_cast<double>(end.tv_usec - start.tv_usec));
#endif

	QString txt;
	QTextStream strm(&txt);
	strm << "Time elasped : " << /*timer.*/elapsed/*()*/ << " ms, iterations: " << niters;
	statusBarLabel->setText(txt);
}
// -------------------------------------------------------------------------
void Morph::clError(const QString& message, cl_int err)
{
	if(err != CL_SUCCESS)
	{
		QMessageBox::critical(this, "OpenCL error", message,
			QMessageBox::Ok);
		exit(1);
	}
}
// -------------------------------------------------------------------------
void Morph::initOpenCL()
{
	// Connect to a compute device
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.empty())
	{
		QMessageBox::critical(nullptr,
			"Critical error",
			"No OpenCL Platform available therefore OpenCL processing will be disabled",
			QMessageBox::Ok);
		ui.actionOpenCL->setChecked(false);
		ui.actionOpenCL->setEnabled(false);	
		return;
	}

	// FIXME
	cl::Platform platform = platforms[0];
	cl_context_properties properties[] = { 
		CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),
		0, 0
	};

        // Tylko GPU (i tak CPU chwilowo AMD uwalil)
	cl_int err;
	context = cl::Context(CL_DEVICE_TYPE_CPU, properties, nullptr, nullptr, &err);
	clError("Failed to create compute context!", err);

	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// FIXME
	dev = devices[0];
	std::vector<cl::Device> devs(1);
	devs[0] = (dev);

	// Kolejka polecen
	cq = cl::CommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	clError("Failed to create command queue!", err);

	// Zaladuj Kernele
	QFile file("./naive-kernels.cl");
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
		clError("Can't read naive-kernels.cl file", -1);

	QTextStream in(&file);
	QString contents = in.readAll();

	QByteArray w = contents.toLocal8Bit();
	const char* src = w.data();
	size_t len = contents.length();

	cl::Program::Sources sources(1, std::make_pair(src, len));
	cl::Program program = cl::Program(context, sources, &err);
	clError("Failed to create compute program!", err);

	err = program.build(devs);
	if(err != CL_SUCCESS)
	{
		QString log(QString::fromStdString(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev)));
		clError(log, -1);
	}

	// Stworz kernele ze zbudowanego programu
	kernelSubtract = cl::Kernel(program, "subtract", &err);
	clError("Failed to create dilate kernel!", err);

	kernelAddHalf = cl::Kernel(program, "addHalf", &err);
	clError("Failed to create addHalf kernel!", err);
	
	kernelErode = cl::Kernel(program, "erode", &err);
	clError("Failed to create erode kernel!", err);

	kernelDilate = cl::Kernel(program, "dilate", &err);
	clError("Failed to create dilate kernel!", err);

	kernelRemove = cl::Kernel(program, "remove", &err);
	clError("Failed to create remove kernel!", err);

	for(int i = 0; i < 8; ++i)
	{
		QString kernelName = "skeleton_iter" + QString::number(i+1);
		QByteArray kk = kernelName.toAscii();
		const char* k = kk.data();
		kernelSkeleton_iter[i] = cl::Kernel(program, k, &err);
		clError("Failed to create skeleton_iter kernel!", err);
	}	

	ui.actionOpenCL->setChecked(true);
}
// -------------------------------------------------------------------------
void Morph::morphologyOpenCL()
{
	cv::Mat element = standardStructuringElement();

	// Bufor docelowy
	cl_int err;
	clDst = cl::Buffer(context,
		CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, 
		src.rows * src.cols, // obraz 1-kanalowy
		nullptr, &err);
	clError("Error while creating destination OpenCL buffer", err);

	// Element strukturalny
	clElement = cl::Buffer(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		element.size().area(), element.ptr<uchar>(), &err);
	clError("Error while creating buffer for structure element!", err);

	int niters = 1;
	cl_ulong elapsed = 0;

	// Erozja
	if(ui.rbErode->isChecked())
	{
		elapsed += executeMorphologyKernel(&kernelErode, clSrc, clDst);
	}
	// Dylatacja
	else if(ui.rbDilate->isChecked())
	{
		elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clDst);
	}
	else
	{
		// Funkcja lambda kopiujaca zawartosc jednego bufora OpenCL'a do drugiego
		// przy okazji mierzac czas tej operacji
		auto copyBuffer = [this](const cl::Buffer& clsrc, cl::Buffer& cldst,
			cl::Event& clevt) -> cl_ulong
		{
			cq.enqueueCopyBuffer(clsrc, cldst, 
				0, 0, src.size().area(), 
				nullptr, &clevt);
			clevt.wait();
			return elapsedEvent(clevt);

		};

		// Potrzebowac bedziemy dodatkowego bufora tymczasowego
		clTmp = cl::Buffer(context,
			CL_MEM_READ_WRITE, 
			src.rows * src.cols, // obraz 1-kanalowy
			nullptr, &err);
		clError("Error while creating temporary OpenCL buffer", err);

		// Otwarcie
		if(ui.rbOpen->isChecked())
		{
			// dst = dilate(erode(src))
			elapsed += executeMorphologyKernel(&kernelErode, clSrc, clTmp);
			elapsed += executeMorphologyKernel(&kernelDilate, clTmp, clDst);
		}
		// Zamkniecie
		else if(ui.rbClose->isChecked())
		{
			// dst = erode(dilate(src))
			elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clTmp);
			elapsed += executeMorphologyKernel(&kernelErode, clTmp, clDst);
		}
		// Operacja wyciagania konturow
		else if(ui.rbRemove->isChecked())
		{
			// Skopiuj obraz zrodlowy do docelowego
			cl::Event evt;
			elapsed += copyBuffer(clSrc, clDst, evt);
			elapsed += executeRemoveKernel(clSrc, clDst);
		}
		// Operacja szkieletyzacji
		else if(ui.rbSkeleton->isChecked())
		{
			// Skopiuj obraz zrodlowy do docelowego
			cl::Event evt;	
			copyBuffer(clSrc, clTmp, evt);
			copyBuffer(clSrc, clDst, evt);

			for(int iters = 0; iters < 111; ++iters)
			{
				for(int i = 0; i < 8; ++i)
				{
					elapsed += executeSkeletonKernel(i, clTmp, clDst);

					// Kopiowanie bufora
					copyBuffer(clDst, clTmp, evt);
					elapsed += elapsedEvent(evt);
				}
			}

			elapsed += executeAddHalfKernel(clSrc, clDst);
		}
		else
		{
			// Potrzebowac bedziemy dodatkowego bufora tymczasowego
			clTmp2 = cl::Buffer(context,
				CL_MEM_READ_WRITE, 
				src.rows * src.cols, // obraz 1-kanalowy
				nullptr, &err);
			clError("Error while creating temporary OpenCL buffer", err);

			// Gradient morfologiczny
			if(ui.rbGradient->isChecked())
			{ 
				//dst = dilate(src) - erode(src);
				elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clTmp);
				elapsed += executeMorphologyKernel(&kernelErode, clSrc, clTmp2);
				elapsed += executeSubtractKernel(clTmp, clTmp2, clDst);
			}
			// TopHat
			else if(ui.rbTopHat->isChecked())
			{ 
				// dst = src - dilate(erode(src))
				elapsed += executeMorphologyKernel(&kernelErode, clSrc, clTmp);
				elapsed += executeMorphologyKernel(&kernelDilate, clTmp, clTmp2);
				elapsed += executeSubtractKernel(clSrc, clTmp2, clDst);
			}
			// BlackHat
			else if(ui.rbBlackHat->isChecked())
			{ 
				// dst = close(src) - src
				elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clTmp);
				elapsed += executeMorphologyKernel(&kernelErode, clTmp, clTmp2);
				elapsed += executeSubtractKernel(clTmp2, clSrc, clDst);
			}
			else
			{
				return;
			}
		}
	}

	// Zczytaj wynik
	cv::Mat dst(src.size(), CV_8U);
	cl::Event evt;
	cq.enqueueReadBuffer(clDst, CL_FALSE, 0,
		src.cols * src.rows, dst.ptr<uchar>(),
		nullptr, &evt);
	evt.wait();

	// Ile czasu zajelo zczytanie danych z powrotem
	elapsed += elapsedEvent(evt);
	// Ile czasu wszystko zajelo
	double delapsed = elapsed * 0.000001;

	showCvImage(dst);

	QString txt; 
	QTextStream strm(&txt);

	strm << "Time elasped : " << delapsed << " ms, iterations: " << niters;
	statusBarLabel->setText(txt);
}
// -------------------------------------------------------------------------
cl_ulong Morph::executeMorphologyKernel(cl::Kernel* kernel, 
	const cl::Buffer& clBufferSrc,
	cl::Buffer& clBufferDst)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernel->setArg(0, clBufferSrc);
	err |= kernel->setArg(1, clBufferDst);
	err |= kernel->setArg(2, clElement);
	err |= kernel->setArg(3, ui.hsXElementSize->value());
	err |= kernel->setArg(4, ui.hsYElementSize->value());
	clError("Error while setting kernel arguments", err);

	//size_t localSize = kernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(dev, &err);
	//clError("Error while retrieving kernel work group info!", err);

	// Odpal kernela
	cl::Event evt;	
	cq.enqueueNDRangeKernel(*kernel,
		cl::NullRange,
		cl::NDRange(src.cols, src.rows),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong Morph::executeSubtractKernel(const cl::Buffer& clBufferA,
	const cl::Buffer& clBufferB, cl::Buffer& clBufferDst)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernelSubtract.setArg(0, clBufferA);
	err |= kernelSubtract.setArg(1, clBufferB);
	err |= kernelSubtract.setArg(2, clBufferDst);
	clError("Error while setting kernel arguments", err);

	// Odpal kernela
	cl::Event evt;	
	cq.enqueueNDRangeKernel(kernelSubtract,
		cl::NullRange,
		cl::NDRange(src.cols * src.rows),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong Morph::executeAddHalfKernel(const cl::Buffer& clBufferSrc,
	cl::Buffer& clBufferDst)
{
	cl_int err;
	err  = kernelAddHalf.setArg(0, clBufferDst);
	err |= kernelAddHalf.setArg(1, clBufferSrc);
	clError("Error while setting kernel arguments", err);

	// Odpal kernela
	cl::Event evt;
	cq.enqueueNDRangeKernel(kernelAddHalf,
		cl::NullRange,
		cl::NDRange(src.cols * src.rows),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong Morph::executeRemoveKernel(const cl::Buffer& clBufferSrc,
	cl::Buffer& clBufferDst)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernelRemove.setArg(0, clBufferSrc);
	err |= kernelRemove.setArg(1, clBufferDst);
	clError("Error while setting kernel arguments", err);

	// Odpal kernela
	cl::Event evt;
	cq.enqueueNDRangeKernel(kernelRemove,
		cl::NullRange,
		cl::NDRange(src.cols, src.rows),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong Morph::executeSkeletonKernel(int i, const cl::Buffer& clBufferSrc,
	cl::Buffer& clBufferDst)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernelSkeleton_iter[i].setArg(0, clBufferSrc);
	err |= kernelSkeleton_iter[i].setArg(1, clBufferDst);
	clError("Error while setting kernel arguments", err);

	// Odpal kernela
	cl::Event evt;
	cq.enqueueNDRangeKernel(kernelSkeleton_iter[i],
		cl::NullRange,
		cl::NDRange(src.cols, src.rows),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong Morph::elapsedEvent(const cl::Event& evt)
{
	cl_ulong eventstart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong eventend = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	return (cl_ulong)(eventend - eventstart);
}
