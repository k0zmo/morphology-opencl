#include "cache16x16.cl"

__constant uint OBJ = 255;
__constant uint BCK = 0;

__kernel void outline(
	__global uint* input,
	__global uint* output,
	const int rowPitch)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	if (input[(gid.x - 1) + (gid.y - 1) * rowPitch] == OBJ &&
		input[(gid.x    ) + (gid.y - 1) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y - 1) * rowPitch] == OBJ &&
		
		input[(gid.x - 1) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * rowPitch] == OBJ &&
		
		input[(gid.x - 1) + (gid.y + 1) * rowPitch] == OBJ &&
		input[(gid.x    ) + (gid.y + 1) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y + 1) * rowPitch] == OBJ)
	{
		output[gid.x + gid.y * rowPitch] = BCK;
	}
}

#define WORK_GROUP_SIZE 16

__kernel __attribute__((reqd_work_group_size(16,16,1)))
void outline_local(
	__global uint* input,
	__global uint* output,
	const int2 imageSize)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	int2 localSize = (int2)(get_local_size(0), get_local_size(1));
	
	__local uint sharedBlock[WORK_GROUP_SIZE+2][WORK_GROUP_SIZE+2];
	
	if(gid.x < imageSize.x && gid.y < imageSize.y)
	{
		// Wczytaj piksel do pamieci lokalnej odpowiadajacy temu w pamieci globalnej
		sharedBlock[lid.y][lid.x] = input[gid.y*imageSize.x + gid.x];
		
		// Wczytaj piksele z 'apron'
		if(lid.x < 2)
		{
			int x = gid.x + localSize.x;
			if(x < imageSize.x)
				sharedBlock[lid.y][lid.x + localSize.x] = input[gid.y*imageSize.x + x];
		}
		
		if(lid.y < 2)
		{
			int y = gid.y + localSize.y;
			if(y < imageSize.y)
				sharedBlock[lid.y + localSize.y][lid.x] = input[y*imageSize.x + gid.x];
		}
		
		if(lid.x > 13 && lid.y > 13)
		{
			int x = gid.x + 2;
			int y = gid.y + 2;
			
			if(x < imageSize.x && y < imageSize.y)
				sharedBlock[lid.y + 2][lid.x + 2] = input[y*imageSize.x + x];
		}
	}		
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;
		
	uint v1 = sharedBlock[lid.y    ][lid.x    ];
	uint v2 = sharedBlock[lid.y    ][lid.x + 1];
	uint v3 = sharedBlock[lid.y    ][lid.x + 2];
	uint v4 = sharedBlock[lid.y + 1][lid.x    ];
	uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v1 == OBJ &&
		v2 == OBJ &&
		v3 == OBJ &&
		v4 == OBJ &&
		v6 == OBJ &&
		v7 == OBJ &&
		v8 == OBJ &&
		v9 == OBJ)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
	}
}

__kernel __attribute__((reqd_work_group_size(16,16,1)))
void outline4_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize)
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
		
	int2 lid = (int2)(get_local_id(0), get_local_id(1));	
	uint v1 = sharedBlock[lid.y    ][lid.x    ];
	uint v2 = sharedBlock[lid.y    ][lid.x + 1];
	uint v3 = sharedBlock[lid.y    ][lid.x + 2];
	uint v4 = sharedBlock[lid.y + 1][lid.x    ];
	uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v1 == OBJ &&
		v2 == OBJ &&
		v3 == OBJ &&
		v4 == OBJ &&
		v6 == OBJ &&
		v7 == OBJ &&
		v8 == OBJ &&
		v9 == OBJ)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
	}
}