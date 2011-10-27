#include "common.cl"

__kernel void subtract(
	__global uint* a,
	__global uint* b,
	__global uint* output)
{
	size_t gid = get_global_id(0);
	output[gid] = sub_sat(a[gid], b[gid]);
}

__kernel void subtract4(
	__global uint4* a,
	__global uint4* b,
	__global uint4* output)
{
	size_t gid = get_global_id(0);
	output[gid] = sub_sat(a[gid], b[gid]);
}