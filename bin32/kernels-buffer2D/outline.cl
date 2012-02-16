#include "common.cl"

__kernel void outline(
	__read_only image2d_t src,
	__write_only image2d_t dst)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{	
		float v1 = read_imagef(src, smp, gid + (int2){-1, -1}).x;
		float v2 = read_imagef(src, smp, gid + (int2){ 0, -1}).x;
		float v3 = read_imagef(src, smp, gid + (int2){ 1, -1}).x;
		float v4 = read_imagef(src, smp, gid + (int2){-1,  0}).x;
		// v5
		float v6 = read_imagef(src, smp, gid + (int2){ 1,  0}).x;
		float v7 = read_imagef(src, smp, gid + (int2){-1,  1}).x;
		float v8 = read_imagef(src, smp, gid + (int2){ 0,  1}).x;
		float v9 = read_imagef(src, smp, gid + (int2){ 1,  1}).x;
		
		if (v1 == OBJ &&
			v2 == OBJ &&
			v3 == OBJ &&
			v4 == OBJ &&
			v6 == OBJ &&
			v7 == OBJ &&
			v8 == OBJ &&
			v9 == OBJ)
		{
			write_imagef(dst, gid, (float4)(BCK));
		}
	}
}