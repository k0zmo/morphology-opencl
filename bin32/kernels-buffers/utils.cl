#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uchar OBJ = 255;
__constant uchar BCK = 0;

__kernel void subtract(
	__global uchar* a,
	__global uchar* b,
	__global uchar* output)
{
	size_t gid = get_global_id(0);
	output[gid] = sub_sat(a[gid], b[gid]);
}

__kernel void subtract4(
	__global uchar4* a,
	__global uchar4* b,
	__global uchar4* output)
{
	size_t gid = get_global_id(0);
	output[gid] = sub_sat(a[gid], b[gid]);
}