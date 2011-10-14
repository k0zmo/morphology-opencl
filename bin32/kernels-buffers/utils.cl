#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uchar OBJ = 255;
__constant uchar BCK = 0;

__kernel void subtract(
	__global uchar* a,
	__global uchar* b,
	__global uchar* output)
{
	size_t gid = get_global_id(0);
#ifdef SUBTRACT_SAT
	output[gid] = sub_sat(a[gid], b[gid]);
#else
	output[gid] = (b[gid] > a[gid]) ? (0): (a[gid] - b[gid]);
#endif
}

__kernel void subtract4(
	__global uchar4* a,
	__global uchar4* b,
	__global uchar4* output)
{
	size_t gid = get_global_id(0);
	output[gid] = sub_sat(a[gid], b[gid]);
}

#ifdef USE_ATOMIC_COUNTERS
#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable 
#else
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#endif

__kernel void diffPixels(
	__global uchar* a,
	__global uchar* b,
#ifdef USE_ATOMIC_COUNTERS
	counter32_t counter)
#else
	__global uint* counter)
#endif
{
	size_t gid = get_global_id(0);
	
	if(a[gid] != b[gid])
		(void) atomic_inc(counter);
}

__kernel void diffPixels4(
	__global uint4* a,
	__global uint4* b,
#ifdef USE_ATOMIC_COUNTERS
	counter32_t counter)
#else
	__global uint* counter)
#endif
{
	int gid = get_global_id(0);

	uint4 v1 = a[gid];
	uint4 v2 = b[gid];
	
	if(v1.x != v2.x) atomic_inc(counter);
	if(v1.y != v2.y) atomic_inc(counter);
	if(v1.z != v2.z) atomic_inc(counter);
	if(v1.w != v2.w) atomic_inc(counter);
}