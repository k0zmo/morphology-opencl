__constant uint erodeINF = 255;
__constant uint dilateINF = 0;
__constant uint OBJ = 255;
__constant uint BCK = 0;

__constant sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;