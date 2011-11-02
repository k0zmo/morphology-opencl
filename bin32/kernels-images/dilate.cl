#include "common.cl"

__kernel void dilate(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant int2* coords,
	const int coords_size)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	uint val = dilateINF;
	
	for(int i = 0; i < coords_size; ++i)
	{
		int2 coord = coords[i] + gid;	
		val = max(val, read_imageui(src, smp, coord).x);
	}
	
	write_imageui(dst, gid, (uint4)(val));
}

__kernel void dilate_c4(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant int4* coords,
	const int coords_size)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	uint val = dilateINF;
	int c2 = coords_size >> 1;
	
	for(int i = 0; i < c2; ++i)
	{
		int4 coord = coords[i] + (int4)(gid, gid);	
		
		val = max(val, read_imageui(src, smp, coord.xy).x);
		val = max(val, read_imageui(src, smp, coord.zw).x);
	}
	
	if(coords_size % 2)
	{
		__constant int2* c = (__constant int2*)(coords);
		int2 coord = c[coords_size-1] + gid;
		val = max(val, read_imageui(src, smp, coord).x);
	}
	
	write_imageui(dst, gid, (uint4)(val));
}

#ifndef COORDS_SIZE
#define COORDS_SIZE 169
#endif

__kernel void dilate_c4_def(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant int4* coords)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	uint val = dilateINF;
	int c2 = COORDS_SIZE >> 1;
	
	#pragma unroll
	for(int i = 0; i < c2; ++i)
	{
		int4 coord = coords[i] + (int4)(gid, gid);	
		
		val = max(val, read_imageui(src, smp, coord.xy).x);
		val = max(val, read_imageui(src, smp, coord.zw).x);
	}
	
	if(COORDS_SIZE % 2)
	{
		__constant int2* c = (__constant int2*)(coords);
		int2 coord = c[COORDS_SIZE-1] + gid;
		val = max(val, read_imageui(src, smp, coord).x);
	}
	
	write_imageui(dst, gid, (uint4)(val));
}