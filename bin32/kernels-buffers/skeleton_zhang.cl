#include "common.cl"

__attribute__((always_inline)) 
uint getCode(
	__global uchar* input, int2 gid, 
	int rowPitch, __constant uint* table)
{
	uchar p1 = input[(gid.x - 1) + (gid.y - 1) * rowPitch];
	uchar p2 = input[(gid.x    ) + (gid.y - 1) * rowPitch];
	uchar p3 = input[(gid.x + 1) + (gid.y - 1) * rowPitch];
	uchar p4 = input[(gid.x - 1) + (gid.y    ) * rowPitch];
	uchar p6 = input[(gid.x + 1) + (gid.y    ) * rowPitch];
	uchar p7 = input[(gid.x - 1) + (gid.y + 1) * rowPitch];
	uchar p8 = input[(gid.x    ) + (gid.y + 1) * rowPitch];
	uchar p9 = input[(gid.x + 1) + (gid.y + 1) * rowPitch];
	
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
	__global uchar* input,
	__global uchar* output,
	const int rowPitch,
	__constant uint* table,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	uchar v = input[gid.x + gid.y * rowPitch];
	
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
	__global uchar* input,
	__global uchar* output,
	const int rowPitch,
	__constant uint* table,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	uchar v = input[gid.x + gid.y * rowPitch];
	
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
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX],
	int2 lid, __constant uint* table)
{
	uchar p1 = sharedBlock[lid.y    ][lid.x    ];
	uchar p2 = sharedBlock[lid.y    ][lid.x + 1];
	uchar p3 = sharedBlock[lid.y    ][lid.x + 2];
	uchar p4 = sharedBlock[lid.y + 1][lid.x    ];
	uchar p6 = sharedBlock[lid.y + 1][lid.x + 2];
	uchar p7 = sharedBlock[lid.y + 2][lid.x    ];
	uchar p8 = sharedBlock[lid.y + 2][lid.x + 1];
	uchar p9 = sharedBlock[lid.y + 2][lid.x + 2];
	
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
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize,
	__constant uint* table,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);	
	
	if (gid.y < imageSize.y - 2 && 
		gid.x < imageSize.x - 2)
	{
		// Pobierz srodkowy piksel z pamieci lokalnej
		uchar v = sharedBlock[lid.y + 1][lid.x + 1];
		
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
}	

__kernel 
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeletonZhang4_pass2_local(
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize,
	__constant uint* table,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);	
	
	if (gid.y < imageSize.y - 2 &&
		gid.x >= imageSize.x - 2)
	{
		// Pobierz srodkowy piksel z pamieci lokalnej
		uchar v = sharedBlock[lid.y + 1][lid.x + 1];
		
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
}
