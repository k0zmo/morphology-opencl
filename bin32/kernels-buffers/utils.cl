#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uchar OBJ = 255;
__constant uchar BCK = 0;

__kernel void subtract(
	__global uchar* a,
	__global uchar* b,
	__global uchar* output)
{
	size_t gid = get_global_id(0);
	output[gid] = (b[gid] > a[gid]) ? (0): (a[gid] - b[gid]);
}

__kernel void diffPixels_g(
	__global uchar* a,
	__global uchar* b,
	__global uint* counter)
{
	size_t gid = get_global_id(0);
	
	if(a[gid] != b[gid])
		(void) atomic_inc(counter);
}

__kernel void diffPixels4_g(
	__global uint4* a,
	__global uint4* b,
	__global uint* counter)
{
	int gid = get_global_id(0);
	gid *= 4;

	__global uint* aa = (__global uint*)(a);
	__global uint* bb = (__global uint*)(b);

	#pragma unroll
	for(int i = 0; i < 4; ++i)
		if(aa[gid+i] != bb[gid+i]) 
			(void) atomic_inc(counter);
}

#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable 

__kernel void diffPixels(
	__global uchar* a,
	__global uchar* b,
	counter32_t counter)
{
	size_t gid = get_global_id(0);
	
	if(a[gid] != b[gid])
		atomic_inc(counter);
}

__kernel void diffPixels4(
	__global uint4* a,
	__global uint4* b,
	counter32_t counter)
{
	int gid = get_global_id(0);
	gid *= 4;

	__global uint* aa = (__global uint*)(a);
	__global uint* bb = (__global uint*)(b);

	#pragma unroll
	for(int i = 0; i < 4; ++i)
		if(aa[gid+i] != bb[gid+i]) 
			atomic_inc(counter);
}