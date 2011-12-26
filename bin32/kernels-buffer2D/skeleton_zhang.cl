#include "common.cl"

__attribute__((always_inline))
uint getCode(
	__read_only image2d_t src, const sampler_t smp, 
	int2 gid, __constant uint* table)
{
	uint p1 = read_imageui(src, smp, gid + (int2){-1, -1}).x;
	uint p2 = read_imageui(src, smp, gid + (int2){ 0, -1}).x;
	uint p3 = read_imageui(src, smp, gid + (int2){ 1, -1}).x;
	uint p4 = read_imageui(src, smp, gid + (int2){-1,  0}).x;
	uint p6 = read_imageui(src, smp, gid + (int2){ 1,  0}).x;
	uint p7 = read_imageui(src, smp, gid + (int2){-1,  1}).x;
	uint p8 = read_imageui(src, smp, gid + (int2){ 0,  1}).x;
	uint p9 = read_imageui(src, smp, gid + (int2){ 1,  1}).x;
	
	// lut index
	uint index = 
		((p4&0x01) << 7) |
		((p7&0x01) << 6) |
		((p8&0x01) << 5) |
		((p9&0x01) << 4) |
		((p6&0x01) << 3) |
		((p3&0x01) << 2) |
		((p2&0x01) << 1) |
		 (p1&0x01);
	return table[index];
}

__kernel void skeletonZhang_pass1(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant uint* table,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{
		uint v = read_imageui(src, smp, gid).x;
		
		if(v != BCK)
		{
			uint code = getCode(src, smp, gid, table);
			
			if(code == 2 || code == 3)
			{
				// pixelRemoved++
				atomic_inc(counter);
				write_imageui(dst, gid, (uint4)(BCK));
			}
		}
	}
}

__kernel void skeletonZhang_pass2(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant uint* table,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{
		uint v = read_imageui(src, smp, gid).x;
		
		if(v != BCK)
		{
			uint code = getCode(src, smp, gid, table);
			
			if(code == 1 || code == 3)
			{
				// pixelRemoved++
				atomic_inc(counter);
				write_imageui(dst, gid, (uint4)(BCK));
			}
		}
	}
}