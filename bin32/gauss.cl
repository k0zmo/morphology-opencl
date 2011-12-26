__constant sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
__kernel void gaussianRow(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__global float* mask,
	int radius)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	float4 rgba = (float4)(0);
	
	for(int k = -radius; k <= radius; ++k)
	{
		int2 coord = gid + (int2){k, 0};
		float m = mask[k + radius];
		rgba += read_imagef(src, smp, coord) * (float4)(m, m, m, 1) ;
	}
	
	write_imagef(dst, gid, rgba);
}

__kernel void gaussianCol(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__global float* mask,
	int radius)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	float4 rgba = (float4)(0);
	
	for(int k = -radius; k <= radius; ++k)
	{
		int2 coord = gid + (int2){0, k};
		float m = mask[k + radius];
		rgba += read_imagef(src, smp, coord) * (float4)(m, m, m, 1) ;
	}
	
	write_imagef(dst, gid, rgba);
}

#ifndef RADIUS
#define RADIUS 3
#endif

__kernel void gaussianRow_pragma(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__global float* mask,
	int radius)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	float4 rgba = (float4)(0);
	
	#pragma unroll
	for(int k = -RADIUS; k <= RADIUS; ++k)
	{
		int2 coord = gid + (int2){k, 0};
		float m = mask[k + RADIUS];
		rgba += read_imagef(src, smp, coord) * (float4)(m, m, m, 1) ;
	}
	
	write_imagef(dst, gid, rgba);
}

__kernel void gaussianCol_pragma(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__global float* mask,
	int radius)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	float4 rgba = (float4)(0);
	
	#pragma unroll
	for(int k = -RADIUS; k <= RADIUS; ++k)
	{
		int2 coord = gid + (int2){0, k};
		float m = mask[k + RADIUS];
		rgba += read_imagef(src, smp, coord) * (float4)(m, m, m, 1) ;
	}
	
	write_imagef(dst, gid, rgba);
}