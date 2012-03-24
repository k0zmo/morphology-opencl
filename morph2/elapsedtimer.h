#pragma once

#if !defined(_WIN32)
	#include <sys/time.h>
#else
	#include <windows.h>
#endif

class ElapsedTimer
{
public:
	void start();
	double elapsed();

private:
	#if defined(_WIN32)
		LARGE_INTEGER mfreq, mstart, mend;
	#else
		timeval mstart, mend;
	#endif
};