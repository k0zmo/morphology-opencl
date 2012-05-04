#include "common.cl"

__kernel void subtract(
	__global type_t* a,
	__global type_t* b,
	__global type_t* output,
	const uint n)
{
	size_t gid = get_global_id(0);
	if (gid < n)
		output[gid] = sub_sat(a[gid], b[gid]);
}

__kernel void subtract4(
	__global type4_t* a,
	__global type4_t* b,
	__global type4_t* output,
	const uint n)
{
	size_t gid = get_global_id(0);
	if (gid < n)
		output[gid] = sub_sat(a[gid], b[gid]);
}