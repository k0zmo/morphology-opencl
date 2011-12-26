#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <cl/cl.hpp>

bool loadFile(std::string& contents, const std::string& filename)
{
	std::ifstream strm;
	strm.open(filename.c_str(), std::ios::binary | std::ios_base::in);
	if(!strm.is_open())
	{
		fprintf(stderr, "Error opening given file.\n");
		return false;
	}
	strm.seekg(0, std::ios::end);
	contents.reserve(static_cast<size_t>(strm.tellg()));
	strm.seekg(0);

	contents.assign(std::istreambuf_iterator<char>(strm),
		std::istreambuf_iterator<char>());
	return true;
}

void clError(const std::string& message, cl_int err)
{
	if(err != CL_SUCCESS)
	{
		std::cerr << message << "\n";
		exit(1);
	}
}

cl_ulong elapsedEvent(const cl::Event& evt)
{
	cl_ulong eventstart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong eventend = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	return (cl_ulong)(eventend - eventstart);
}

int main(int argc, char** argv)
{
	if(argc == 1)
	{
		printf("gauss [flags] input\n"
		       "Flags:\n"
			   "  --opencv (default is opencl)\n"
			   "  --maxradius <N>\n"
			   "  --pragma (will unroll and recompile)\n");
		exit(-1);
	}

	int maxradius = 1;
	bool useopencl = true;
	std::string input;
	bool recompile = false;

	for(int i = 1; i < argc; ++i)
	{
		if(!strcmp(argv[i], "--opencv"))
		{
			useopencl = false;
		}
		else if(!strcmp(argv[i], "--maxradius"))
		{
			sscanf(argv[i+1], "%d", &maxradius);
			++i;
		}
		else if(!strcmp(argv[i], "--pragma"))
		{
			recompile = true;
		}
		else
		{
			input = argv[i];
		}
	}

	// Read source image
	cv::Mat sourceImage = cv::imread(input);
	cv::Mat dstImage(sourceImage.size(), CV_8UC4);

	// Add 4th channel
	std::vector<cv::Mat> mv;
	cv::split(sourceImage, mv);
	mv.emplace_back(cv::Mat(sourceImage.size(), CV_8U, cv::Scalar(255)));
	cv::merge(mv, sourceImage);
	assert(sourceImage.type() == CV_8UC4);

	std::fstream fout("output.txt", std::ios::out);

	if(useopencl)
	{
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		if(platforms.size() == 0)
		{
			std::cout << "No OpenCL Platform available\n";
			exit(1);
		}

		cl::Platform platform = platforms[0];
		cl_context_properties properties[] = { 
			CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),
			0, 0
		};

		cl_int err;
		cl::Context context(CL_DEVICE_TYPE_ALL, properties, nullptr, nullptr, &err);
		clError("Failed to create compute context!", err);

		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		cl::Device dev;

		if(devices.empty()) 
		{
			std::cerr << "No devices for chosen platform!";
			exit(1);

		}
		else if(devices.size() == 1)
		{
			auto deviceName = devices[0].getInfo<CL_DEVICE_NAME>();
			auto deviceType = devices[0].getInfo<CL_DEVICE_TYPE>();

			std::cout << "Device: " << deviceName << " (" <<
				((deviceType == CL_DEVICE_TYPE_CPU) ? "CPU" : "GPU") << ")\n";
			dev = devices[0];
		}
		else
		{
			std::cout << "Available devices:\n";

			int i = 1;
			for (auto it = devices.begin(), ite = devices.end(); it != ite; ++it)
			{
				auto deviceName = it->getInfo<CL_DEVICE_NAME>();
				auto deviceType = it->getInfo<CL_DEVICE_TYPE>();

				std::cout << "(" << i++ << ") Device: " << deviceName << " (" <<
					((deviceType == CL_DEVICE_TYPE_CPU) ? "CPU" : "GPU") << ")\n";
			}

			int c = 0;
			while (c <= 0 || static_cast<size_t>(c) > devices.size())
			{
				std::cout << "Pick a device: ";
				std::cin >> c;
			}		

			dev = devices[c-1];
		}

		std::vector<cl::Device> devs(1);
		devs[0] = (dev);

		// Create command queue
		cl::CommandQueue cq(context, dev, CL_QUEUE_PROFILING_ENABLE, &err);
		clError("Failed to create command queue!", err);

		cl::Image2D src(context, CL_MEM_READ_ONLY, 
			cl::ImageFormat(CL_BGRA, CL_UNORM_INT8), 
			sourceImage.cols, sourceImage.rows, 0, 0, &err);
		clError("Failed to create source image!", err);

		cl::Image2D tmp(context, CL_MEM_READ_WRITE, 
			cl::ImageFormat(CL_BGRA, CL_UNORM_INT8),
			sourceImage.cols, sourceImage.rows,
			0, 0, &err);
		clError("Failed to create source image!", err);

		cl::Image2D dst(context, CL_MEM_WRITE_ONLY, 
			cl::ImageFormat(CL_BGRA, CL_UNORM_INT8),
			sourceImage.cols, sourceImage.rows,
			0, 0, &err);
		clError("Failed to create source image!", err);

		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = sourceImage.cols;
		region[1] = sourceImage.rows;
		region[2] = 1;

		size_t row_pitch = sourceImage.step1();
		err = cq.enqueueWriteImage(src, CL_TRUE, 
			origin, region, row_pitch, 0, 
			sourceImage.ptr<uchar*>());

		auto getCompiledKernels = [&](cl::Kernel& kernel_GaussianRow, 
			cl::Kernel& kernel_GaussianCol,
			int radius)
		{
			// Load kernel file
			std::string kernel_src;
			loadFile(kernel_src, "gauss.cl");

			// Create program
			cl::Program::Sources sources(1, std::make_pair(kernel_src.c_str(), kernel_src.size()));
			cl::Program program = cl::Program(context, sources, &err);
			clError("Failed to create compute program!", err);

			char opts[32];
			sprintf(opts, "-DRADIUS=%d", radius);
			err = program.build(devs, opts);
			if(err != CL_SUCCESS)
			{
				auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
				std::cerr << log << "\n";
				exit(1);
			}

			// Create kernel from the program
			kernel_GaussianRow = cl::Kernel(program,
				(recompile ? "gaussianRow_pragma" : "gaussianRow"), &err);
			clError("Failed to create kernel!", err);

			// Create kernel from the program
			kernel_GaussianCol = cl::Kernel(program,
				(recompile ?  "gaussianCol_pragma" : "gaussianRow"), &err);
			clError("Failed to create kernel!", err);	
		};

		cl::Kernel kernel_GaussianRow, kernel_GaussianCol;
		if(!recompile)
			getCompiledKernels(kernel_GaussianRow, kernel_GaussianCol, 3);

		for(int radius = 1; radius <= maxradius; ++radius)
		{
			cv::Size ksize(2*radius+1, 2*radius+1);
			double sigma = 1.5;

			cv::Mat gkernel = cv::getGaussianKernel(ksize.width, sigma, CV_32F);
			float* kernelptr = (float*)gkernel.ptr();

			cl::Buffer bKernel(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				(2*radius+1) * sizeof(float), kernelptr);

			std::cout << "\nSize: " << 2*radius+1 << "x" << 2*radius+1 << ":\n";

			if(recompile)
				getCompiledKernels(kernel_GaussianRow, kernel_GaussianCol, radius);

			for (int i = 0; i < 5; ++i)
			{
				cl_int err = 0;
				err |= kernel_GaussianRow.setArg(0, src);
				err |= kernel_GaussianRow.setArg(1, tmp);
				err |= kernel_GaussianRow.setArg(2, bKernel);
				err |= kernel_GaussianRow.setArg(3, radius);

				cl::Event evt;
				err |= cq.enqueueNDRangeKernel(kernel_GaussianRow, 
					cl::NullRange,
					cl::NDRange(sourceImage.cols, sourceImage.rows),
					cl::NullRange, 0, &evt);
				evt.wait();
				cl_ulong elapsed1 = elapsedEvent(evt);

				err |= kernel_GaussianCol.setArg(0, tmp);
				err |= kernel_GaussianCol.setArg(1, dst);
				err |= kernel_GaussianCol.setArg(2, bKernel);
				err |= kernel_GaussianCol.setArg(3, radius);

				err |= cq.enqueueNDRangeKernel(kernel_GaussianCol,
					cl::NullRange,
					cl::NDRange(sourceImage.cols, sourceImage.rows),
					cl::NullRange, 0, &evt);
				evt.wait();
				cl_ulong elapsed2 = elapsedEvent(evt);

				double totalElapsed = (elapsed1 + elapsed2) * 0.000001;
				std::cout << totalElapsed << "(" << elapsed2 * 0.000001 << " + " << elapsed1 * 0.000001<< ")" << std::endl;
				fout << totalElapsed << std::endl;

				cq.enqueueReadImage(dst, CL_TRUE, 
					origin,	region,
					dstImage.step1(), 0, dstImage.ptr());
			}
		}
	}
	////////////////////////////////////////////////////////////////////////////////////////////
	else
	{
		for(int radius = 1; radius <= maxradius; ++radius)
		{
			cv::Size ksize(2*radius+1, 2*radius+1);
			double sigma = 1.5;

			cv::Mat gkernel = cv::getGaussianKernel(ksize.width, sigma, CV_32F);
			float* kernelptr = (float*)gkernel.ptr();

			std::cout << "\nSize: " << 2*radius+1 << "x" << 2*radius+1 << ":\n";

			for (int i = 0; i < 5; ++i)
			{
				LARGE_INTEGER freq, start, end;
				QueryPerformanceFrequency(&freq);
				QueryPerformanceCounter(&start);

				cv::GaussianBlur(sourceImage, dstImage, ksize, sigma, sigma);

				QueryPerformanceCounter(&end);
				double elapsed = (static_cast<double>(end.QuadPart - start.QuadPart) / 
					static_cast<double>(freq.QuadPart)) * 1000.0f;

				fout << elapsed << std::endl;
				std::cout << elapsed << std::endl;
			}
		}
	}

	fout.close();

	cv::imshow("", dstImage);
	cv::imwrite("output.png", dstImage);
	cv::waitKey(0);
}