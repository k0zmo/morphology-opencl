#include "hitmiss_common.cl"

__attribute__((always_inline))
uint getCode(
	__read_only image2d_t src,
	const sampler_t smp, 
	int2 gid,
	__constant uint* table)
{
	float p1 = read_imagef(src, smp, gid + (int2){-1, -1}).x;
	float p2 = read_imagef(src, smp, gid + (int2){ 0, -1}).x;
	float p3 = read_imagef(src, smp, gid + (int2){ 1, -1}).x;
	float p4 = read_imagef(src, smp, gid + (int2){-1,  0}).x;
	float p6 = read_imagef(src, smp, gid + (int2){ 1,  0}).x;
	float p7 = read_imagef(src, smp, gid + (int2){-1,  1}).x;
	float p8 = read_imagef(src, smp, gid + (int2){ 0,  1}).x;
	float p9 = read_imagef(src, smp, gid + (int2){ 1,  1}).x;

	// lut index
	float4 p1234 = { p1, p2, p3, p4 };
	float4 p6789 = { p6, p7, p8, p9 };
	p1234 *= (float4) { 1.0f, 2.0f, 4.0f, 128.0f };
	p6789 *= (float4) { 8.0f, 64.0f, 32.0f, 16.0f };
	
	p1234.xy += p1234.zw;
	p1234.x += p1234.y;
	
	p6789.xy += p6789.zw;
	p6789.x += p6789.y;
	
	uint index = (uint)(p1234.x + p6789.x);

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
		float v = read_imagef(src, smp, gid).x;
		
		if(v != BCK)
		{
			uint code = getCode(src, smp, gid, table);
			
			if(code == 2 || code == 3)
			{
				// pixelRemoved++
				atomic_inc(counter);
				write_imagef(dst, gid, (float4)(BCK));
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
		float v = read_imagef(src, smp, gid).x;
		
		if(v != BCK)
		{
			uint code = getCode(src, smp, gid, table);
			
			if(code == 1 || code == 3)
			{
				// pixelRemoved++
				atomic_inc(counter);
				write_imagef(dst, gid, (float4)(BCK));
			}
		}
	}
}