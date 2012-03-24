#include "elapsedtimer.h"

void ElapsedTimer::start()
{
#if defined(_WIN32)
	QueryPerformanceFrequency(&mfreq);
	QueryPerformanceCounter(&mstart);
#else
	gettimeofday(&mstart, NULL);
#endif
}

double ElapsedTimer::elapsed()
{
#if defined(_WIN32)
	QueryPerformanceCounter(&mend);
	double elapsed = (static_cast<double>(mend.QuadPart - mstart.QuadPart) / 
		static_cast<double>(mfreq.QuadPart)) * 1000.0f;
#else
	gettimeofday(&mend, NULL);
	double elapsed = (static_cast<double>(mend.tv_sec - mstart.tv_sec) * 1000 +
		0.001f * static_cast<double>(mend.tv_usec - mstart.tv_usec));
#endif

	return elapsed;
}