__constant uchar OBJ = 255;
__constant uchar BCK = 0;

// dst = saturate((int)a - (int)b)

__kernel void subtract(
	__read_only image2d_t a,
	__read_only image2d_t b,
	__write_only image2d_t dst)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
	//int xsize = get_image_width(a);
	//int ysize = get_image_height(a);
	
	//if(gx >= xsize || gy >= ysize)
	//	return;
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
	int2 coords = (int2)(gx, gy);
		
	uchar pixa = read_imageui(a, smp, coords).x;
	uchar pixb = read_imageui(b, smp, coords).x;
	
#ifdef SUBTRACT_SAT
	uchar pix = sub_sat(pixa, pixb);
#else
	uchar pix = (pixb > pixa) ? (0) : (pixa - pixb);
#endif
	
	write_imageui(dst, coords, pix);
}

#ifdef USE_ATOMIC_COUNTERS
#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable 
#else
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

__kernel void diffPixels(
	__read_only image2d_t a,
	__read_only image2d_t b,
#ifdef USE_ATOMIC_COUNTERS
	counter32_t counter)
#else
	__global uint* counter)
#endif
{
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
		
	uchar pixa = read_imageui(a, smp, coords).x;
	uchar pixb = read_imageui(b, smp, coords).x;
	
	if(pixa != pixb)
		(void) atomic_inc(counter);
}

__kernel void diffPixels4(
	__read_only image2d_t a,
	__read_only image2d_t b,
#ifdef USE_ATOMIC_COUNTERS
	counter32_t counter)
#else
	__global uint* counter)
#endif
{
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
	int2 coords = (int2)(get_global_id(0), get_global_id(1));
		
	uint4 pixa = read_imageui(a, smp, coords);
	uint4 pixb = read_imageui(b, smp, coords);

	if(pixa.x != pixb.x) atomic_inc(counter);
	if(pixa.y != pixb.y) atomic_inc(counter);
	if(pixa.z != pixb.z) atomic_inc(counter);
	if(pixa.w != pixb.w) atomic_inc(counter);
}