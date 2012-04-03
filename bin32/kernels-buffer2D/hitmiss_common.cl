#ifndef __HITMISS_COMMON_CL__
#define __HITMISS_COMMON_CL__

#include "common.cl"

#ifdef USE_ATOMIC_COUNTERS
	#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable
	#define counter_type counter32_t
#else
	#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
	#define counter_type volatile __global int*
#endif

#endif
