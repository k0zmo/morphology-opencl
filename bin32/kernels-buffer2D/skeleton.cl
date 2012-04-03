#include "hitmiss_common.cl"

__kernel void skeleton_iter1(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{	
		float v1 = read_imagef(src, smp, gid + (int2){-1, -1}).x;
		float v2 = read_imagef(src, smp, gid + (int2){ 0, -1}).x;
		float v3 = read_imagef(src, smp, gid + (int2){ 1, -1}).x;
		float v5 = read_imagef(src, smp, gid + (int2){ 0,  0}).x;
		float v7 = read_imagef(src, smp, gid + (int2){-1,  1}).x;
		float v8 = read_imagef(src, smp, gid + (int2){ 0,  1}).x;
		float v9 = read_imagef(src, smp, gid + (int2){ 1,  1}).x;
		
		// Element strukturalny pierwszy
		// 1|1|1
		// X|1|X
		// 0|0|0
		//
				
		if (v1 == OBJ &&
			v2 == OBJ &&
			v3 == OBJ &&
			v5 == OBJ &&
			v7 == BCK &&
			v8 == BCK &&
			v9 == BCK)
		{
			write_imagef(dst, gid, (float4)(BCK));
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter2(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{
		float v1 = read_imagef(src, smp, gid + (int2){-1, -1}).x;
		float v3 = read_imagef(src, smp, gid + (int2){ 1, -1}).x;
		float v4 = read_imagef(src, smp, gid + (int2){-1,  0}).x;
		float v5 = read_imagef(src, smp, gid + (int2){ 0,  0}).x;
		float v6 = read_imagef(src, smp, gid + (int2){ 1,  0}).x;
		float v7 = read_imagef(src, smp, gid + (int2){-1,  1}).x;
		float v9 = read_imagef(src, smp, gid + (int2){ 1,  1}).x;
			
		// Element strukturalny pierwszy - 90 w lewo
		// 1|X|0
		// 1|1|0
		// 1|x|0
		//
		
		if (v1 == OBJ &&
			v3 == BCK &&
			v4 == OBJ &&
			v5 == OBJ &&
			v6 == BCK &&
			v7 == OBJ &&
			v9 == BCK)
		{
			write_imagef(dst, gid, (float4)(BCK));
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter3(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{	
		float v1 = read_imagef(src, smp, gid + (int2){-1, -1}).x;
		float v2 = read_imagef(src, smp, gid + (int2){ 0, -1}).x;
		float v3 = read_imagef(src, smp, gid + (int2){ 1, -1}).x;
		float v5 = read_imagef(src, smp, gid + (int2){ 0,  0}).x;
		float v7 = read_imagef(src, smp, gid + (int2){-1,  1}).x;
		float v8 = read_imagef(src, smp, gid + (int2){ 0,  1}).x;
		float v9 = read_imagef(src, smp, gid + (int2){ 1,  1}).x;

		// Element strukturalny pierwszy - 180 w lewo
		// 0|0|0
		// X|1|X
		// 1|1|1
		//
			
		if (v1 == BCK &&
			v2 == BCK &&
			v3 == BCK &&
			v5 == OBJ &&
			v7 == OBJ &&
			v8 == OBJ &&
			v9 == OBJ)
		{
			write_imagef(dst, gid, (float4)(BCK));
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter4(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{	
		float v1 = read_imagef(src, smp, gid + (int2){-1, -1}).x;
		float v3 = read_imagef(src, smp, gid + (int2){ 1, -1}).x;
		float v4 = read_imagef(src, smp, gid + (int2){-1,  0}).x;
		float v5 = read_imagef(src, smp, gid + (int2){ 0,  0}).x;
		float v6 = read_imagef(src, smp, gid + (int2){ 1,  0}).x;
		float v7 = read_imagef(src, smp, gid + (int2){-1,  1}).x;
		float v9 = read_imagef(src, smp, gid + (int2){ 1,  1}).x;

		// Element strukturalny pierwszy - 270 w lewo
		// 0|X|1
		// 0|1|1
		// 0|X|1
		//
		
		if (v1 == BCK &&
			v3 == OBJ &&
			v4 == BCK &&
			v5 == OBJ &&
			v6 == OBJ &&
			v7 == BCK &&
			v9 == OBJ)
		{
			write_imagef(dst, gid, (float4)(BCK));
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter5(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{	
		float v2 = read_imagef(src, smp, gid + (int2){ 0, -1}).x;
		float v4 = read_imagef(src, smp, gid + (int2){-1,  0}).x;
		float v5 = read_imagef(src, smp, gid + (int2){ 0,  0}).x;
		float v6 = read_imagef(src, smp, gid + (int2){ 1,  0}).x;
		float v7 = read_imagef(src, smp, gid + (int2){-1,  1}).x;
		float v8 = read_imagef(src, smp, gid + (int2){ 0,  1}).x;
		
		// Element strukturalny drugi
		// X|1|X
		// 0|1|1
		// 0|0|X
		//

		if (v2 == OBJ &&
			v4 == BCK &&
			v5 == OBJ &&
			v6 == OBJ &&
			v7 == BCK &&
			v8 == BCK)
		{
			write_imagef(dst, gid, (float4)(BCK));
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter6(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{	
		float v2 = read_imagef(src, smp, gid + (int2){ 0, -1}).x;
		float v4 = read_imagef(src, smp, gid + (int2){-1,  0}).x;
		float v5 = read_imagef(src, smp, gid + (int2){ 0,  0}).x;
		float v6 = read_imagef(src, smp, gid + (int2){ 1,  0}).x;
		float v8 = read_imagef(src, smp, gid + (int2){ 0,  1}).x;
		float v9 = read_imagef(src, smp, gid + (int2){ 1,  1}).x;
			
		// Element strukturalny drugi - 90 stopni w lewo
		// X|1|X
		// 1|1|0
		// X|0|0
		//
			
		if (v2 == OBJ &&
			v4 == OBJ &&
			v5 == OBJ &&
			v6 == BCK &&
			v8 == BCK &&
			v9 == BCK)
		{
			write_imagef(dst, gid, (float4)(BCK));
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter7(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{		
		float v2 = read_imagef(src, smp, gid + (int2){ 0, -1}).x;
		float v3 = read_imagef(src, smp, gid + (int2){ 1, -1}).x;
		float v4 = read_imagef(src, smp, gid + (int2){-1,  0}).x;
		float v5 = read_imagef(src, smp, gid + (int2){ 0,  0}).x;
		float v6 = read_imagef(src, smp, gid + (int2){ 1,  0}).x;
		float v8 = read_imagef(src, smp, gid + (int2){ 0,  1}).x;
			
		// Element strukturalny drugi - 180 stopni w lewo
		// X|0|0
		// 1|1|0
		// X|1|X
		//
			
		if (v2 == BCK &&
			v3 == BCK &&
			v4 == OBJ &&
			v5 == OBJ &&
			v6 == BCK &&
			v8 == OBJ)
		{
			write_imagef(dst, gid, (float4)(BCK));
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter8(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 size = { get_image_width(src), get_image_height(src) };
	
	if (all(gid < size))
	{	
		float v1 = read_imagef(src, smp, gid + (int2){-1, -1}).x;
		float v2 = read_imagef(src, smp, gid + (int2){ 0, -1}).x;
		float v4 = read_imagef(src, smp, gid + (int2){-1,  0}).x;
		float v5 = read_imagef(src, smp, gid + (int2){ 0,  0}).x;
		float v6 = read_imagef(src, smp, gid + (int2){ 1,  0}).x;
		float v8 = read_imagef(src, smp, gid + (int2){ 0,  1}).x;
			
		// Element strukturalny drugi - 270 stopni w lewo
		// 0|0|X
		// 0|1|1
		// X|1|X
		//
			
		if (v1 == BCK &&
			v2 == BCK &&
			v4 == BCK &&
			v5 == OBJ &&
			v6 == OBJ &&
			v8 == OBJ)
		{
			write_imagef(dst, gid, (float4)(BCK));
			atomic_inc(counter);
		}
	}
}