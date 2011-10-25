#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#ifdef USE_ATOMIC_COUNTERS
#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable 
#define counter_type counter32_t
#else
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define counter_type __global uint*
#define atomic_inc atom_inc
#endif

#include "cache16x16.cl"

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
			output[gid.x+ + gid.y * rowPitch] = BCK;
		}
	}
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
			output[gid.x+ + gid.y * rowPitch] = BCK;
		}
	}
}

// ###########################################################################

__attribute__((always_inline)) 
uint getCode_local(
	__local uint sharedBlock[SHARED_SIZEY][SHARED_SIZEX],
	int2 lid, __constant uint* table)
{
	uint p1 = sharedBlock[lid.y    ][lid.x    ];
	uint p2 = sharedBlock[lid.y    ][lid.x + 1];
	uint p3 = sharedBlock[lid.y    ][lid.x + 2];
	uint p4 = sharedBlock[lid.y + 1][lid.x    ];
	uint p6 = sharedBlock[lid.y + 1][lid.x + 2];
	uint p7 = sharedBlock[lid.y + 2][lid.x    ];
	uint p8 = sharedBlock[lid.y + 2][lid.x + 1];
	uint p9 = sharedBlock[lid.y + 2][lid.x + 2];
	
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

__kernel 
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeletonZhang4_pass1_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	__constant uint* table,
	counter_type counter)
{
	__local uint sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cacheNeighbours(input, imageSize, sharedBlock);
	
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;
	
	// Pobierz srodkowy piksle z pamieci lokalnej
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	uint v = sharedBlock[lid.y + 1][lid.x + 1];
	
	if(v != BCK)
	{
		// LUT
		uint code = getCode_local(sharedBlock, lid, table);
		
		if(code == 2 || code == 3)
		{
			// pixelRemoved++
			atomic_inc(counter);
			output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		}
	}
}	

__kernel 
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeletonZhang4_pass2_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	__constant uint* table,
	counter_type counter)
{
	__local uint sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cacheNeighbours(input, imageSize, sharedBlock);
	
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;
	
	// Pobierz srodkowy piksle z pamieci lokalnej
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	uint v = sharedBlock[lid.y + 1][lid.x + 1];
	
	if(v != BCK)
	{
		// LUT
		uint code = getCode_local(sharedBlock, lid, table);
		
		if(code == 1 || code == 3)
		{
			// pixelRemoved++
			atomic_inc(counter);
			output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		}
	}
}

