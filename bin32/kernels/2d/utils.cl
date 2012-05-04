#include "common.cl"

// dst = saturate((int)a - (int)b)
// dla CL_UNORM_INT8 dostajemy saturacje za friko

__kernel void subtract(
	__read_only image2d_t a,
	__read_only image2d_t b,
	__write_only image2d_t dst)
{
	int2 coords = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(dst), get_image_height(dst) };
	
	if (all(coords < size))
	{
		float aa = read_imagef(a, smp, coords).x;
		float bb = read_imagef(b, smp, coords).x;
		
		write_imagef(dst, coords, (float4)(aa - bb));
	}
}