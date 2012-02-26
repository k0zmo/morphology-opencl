#ifndef __COMMON_CL__
#define __COMMON_CL__

__constant float erodeINF = 1.0f;
__constant float dilateINF = 0.0f;
__constant float OBJ = 1.0f;
__constant float BCK = 0;

__constant sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
#ifdef USE_ATOMIC_COUNTERS
	#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable 
	#define counter_type counter32_t
#else
	#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
	#define counter_type volatile __global int*
#endif

#endif
