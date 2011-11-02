#include "common.cl"

// dst = saturate((int)a - (int)b)
__kernel void subtract(
	__read_only image2d_t a,
	__read_only image2d_t b,
	__write_only image2d_t dst)
{
	int2 coords = { get_global_id(0), get_global_id(1) };
		
	uint pix = sub_sat(
		read_imageui(a, smp, coords).x,
		read_imageui(b, smp, coords).x);
	
	write_imageui(dst, coords, pix);
}