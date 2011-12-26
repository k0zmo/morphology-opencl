#include "common.cl"

__kernel void subtract(
	__global uchar* a,
	__global uchar* b,
	__global uchar* output,
	const uint n)
{
	size_t gid = get_global_id(0);
	if (gid < n)
		output[gid] = sub_sat(a[gid], b[gid]);
}

__kernel void subtract4(
	__global uchar4* a,
	__global uchar4* b,
	__global uchar4* output,
	const uint n)
{
	size_t gid = get_global_id(0);
	if (gid < n)
		output[gid] = sub_sat(a[gid], b[gid]);
}