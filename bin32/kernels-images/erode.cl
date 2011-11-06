#include "common.cl"

__kernel void erode(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant int2* coords,
	const int coords_size)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{
		uint val = erodeINF;
		
		for(int i = 0; i < coords_size; ++i)
		{
			int2 coord = coords[i] + gid;	
			val = min(val, read_imageui(src, smp, coord).x);
		}
		
		write_imageui(dst, gid, (uint4)(val));
	}
}

__kernel void erode_c4(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant int4* coords,
	const int coords_size)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{
		uint val = erodeINF;
		int c2 = coords_size >> 1;
		
		for(int i = 0; i < c2; ++i)
		{
			int4 coord = coords[i] + (int4)(gid, gid);	
			
			val = min(val, read_imageui(src, smp, coord.xy).x);
			val = min(val, read_imageui(src, smp, coord.zw).x);
		}
		
		if(coords_size % 2)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[coords_size-1] + gid;
			val = min(val, read_imageui(src, smp, coord).x);
		}
		
		write_imageui(dst, gid, (uint4)(val));
	}
}

__kernel void erode_c4_unroll(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant int4* coords,
	const int coords_size)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{
		uint val = erodeINF;
		int c2 = coords_size >> 1;
		int i = 0;
		
		for(; i < c2; i += 2)
		{
			int4 coord0 = coords[i]   + (int4)(gid, gid);
			int4 coord1 = coords[i+1] + (int4)(gid, gid);	
			
			val = min(val, read_imageui(src, smp, coord0.xy).x);
			val = min(val, read_imageui(src, smp, coord0.zw).x);
			val = min(val, read_imageui(src, smp, coord1.xy).x);
			val = min(val, read_imageui(src, smp, coord1.zw).x);
		}
		
		i *= 2;
		
		for( ; i < coords_size; ++i)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[i] + gid;
			val = min(val, read_imageui(src, smp, coord).x);
		}
		
		write_imageui(dst, gid, (uint4)(val));	
	}
}

#ifndef COORDS_SIZE
#define COORDS_SIZE 4
#endif

__kernel void erode_c4_pragma(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant int4* coords,
	const int coords_size /* dummy */)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{
		uint val = erodeINF;
		int c2 = COORDS_SIZE >> 1;
		
		#pragma unroll
		for(int i = 0; i < c2; ++i)
		{
			int4 coord = coords[i] + (int4)(gid, gid);	
			
			val = min(val, read_imageui(src, smp, coord.xy).x);
			val = min(val, read_imageui(src, smp, coord.zw).x);
		}
		
		if(COORDS_SIZE % 2)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[COORDS_SIZE-1] + gid;
			val = min(val, read_imageui(src, smp, coord).x);
		}
		
		write_imageui(dst, gid, (uint4)(val));
	}
}