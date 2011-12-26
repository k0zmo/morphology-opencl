#include "common.cl"

__kernel void outline(
	__read_only image2d_t src,
	__write_only image2d_t dst)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{	
		uint v1 = read_imageui(src, smp, gid + (int2){-1, -1}).x;
		uint v2 = read_imageui(src, smp, gid + (int2){ 0, -1}).x;
		uint v3 = read_imageui(src, smp, gid + (int2){ 1, -1}).x;
		uint v4 = read_imageui(src, smp, gid + (int2){-1,  0}).x;
		// v5
		uint v6 = read_imageui(src, smp, gid + (int2){ 1,  0}).x;
		uint v7 = read_imageui(src, smp, gid + (int2){-1,  1}).x;
		uint v8 = read_imageui(src, smp, gid + (int2){ 0,  1}).x;
		uint v9 = read_imageui(src, smp, gid + (int2){ 1,  1}).x;
		
		if (v1 == OBJ &&
			v2 == OBJ &&
			v3 == OBJ &&
			v4 == OBJ &&
			v6 == OBJ &&
			v7 == OBJ &&
			v8 == OBJ &&
			v9 == OBJ)
		{
			write_imageui(dst, gid, (uint4)(BCK));
		}
	}
}