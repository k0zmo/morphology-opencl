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

#endif
