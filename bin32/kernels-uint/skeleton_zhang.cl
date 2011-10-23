#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#ifdef USE_ATOMIC_COUNTERS
#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable 
#define counter_type counter32_t
#else
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define counter_type __global uint*
#define atomic_inc atom_inc
#endif

__constant uint OBJ = 255;
__constant uint BCK = 0;

__attribute__((always_inline)) 
uint getCode(
	__global uint* input, int2 gid, 
	int rowPitch, __constant uint* table)
{
	uint p1 = input[(gid.x - 1) + (gid.y - 1) * rowPitch];
	uint p2 = input[(gid.x    ) + (gid.y - 1) * rowPitch];
	uint p3 = input[(gid.x + 1) + (gid.y - 1) * rowPitch];
	uint p4 = input[(gid.x - 1) + (gid.y    ) * rowPitch];
	uint p6 = input[(gid.x + 1) + (gid.y    ) * rowPitch];
	uint p7 = input[(gid.x - 1) + (gid.y + 1) * rowPitch];
	uint p8 = input[(gid.x    ) + (gid.y + 1) * rowPitch];
	uint p9 = input[(gid.x + 1) + (gid.y + 1) * rowPitch];
	
	// lut index
	uint index = 
		((p4&0x01) << 7) |
		((p7&0x01) << 6) |
		((p8&0x01) << 5) |
		((p9&0x01) << 4) |
		((p6&0x01) << 3) |
		((p3&0x01) << 2) |
		((p2&0x01) << 1) |
		 (p1&0x01);
	return table[index];
}

__kernel void skeletonZhang_pass1(
	__global uint* input,
	__global uint* output,
	const int rowPitch,
	__constant uint* table,
	counter_type counter)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	uint v = input[gid.x + gid.y * rowPitch];
	
	if(v != BCK)
	{
		uint code = getCode(input, gid, rowPitch, table);
		
		if(code == 2 || code == 3)
		{
			// pixelRemoved++
			atomic_inc(counter);
			v = BCK;
		}
	}
	
	output[(gid.x+1) + (gid.y+1) * rowPitch] = v;
}

__kernel void skeletonZhang_pass2(
	__global uint* input,
	__global uint* output,
	const int rowPitch,
	__constant uint* table,
	counter_type counter)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	uint v = input[gid.x + gid.y * rowPitch];
	
	if(v != BCK)
	{
		uint code = getCode(input, gid, rowPitch, table);
		
		if(code == 1 || code == 3)
		{
			// pixelRemoved++
			atomic_inc(counter);
			v = BCK;
		}
	}
	
	output[(gid.x+1) + (gid.y+1) * rowPitch] = v;
}