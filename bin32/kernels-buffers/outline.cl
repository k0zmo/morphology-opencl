#include "common.cl"

__kernel void outline(
	__global uchar* input,
	__global uchar* output,
	const int rowPitch)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	
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

__kernel
__attribute__((reqd_work_group_size(16,16,1)))
void outline_local(
	__global uchar* input,
	__global uchar* output,
	const int2 imageSize)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	#define WORK_GROUP_SIZE 16
	__local uchar sharedBlock[WORK_GROUP_SIZE+2][WORK_GROUP_SIZE+2];
	
	if(gid.x < imageSize.x && gid.y < imageSize.y)
	{
		// Wczytaj piksel do pamieci lokalnej odpowiadajacy temu w pamieci globalnej
		sharedBlock[lid.y][lid.x] = input[gid.y*imageSize.x + gid.x];
		int2 localSize = { get_local_size(0), get_local_size(1) };
		
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
	
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		uchar v1 = sharedBlock[lid.y    ][lid.x    ];
		uchar v2 = sharedBlock[lid.y    ][lid.x + 1];
		uchar v3 = sharedBlock[lid.y    ][lid.x + 2];
		uchar v4 = sharedBlock[lid.y + 1][lid.x    ];
		uchar v6 = sharedBlock[lid.y + 1][lid.x + 2];
		uchar v7 = sharedBlock[lid.y + 2][lid.x    ];
		uchar v8 = sharedBlock[lid.y + 2][lid.x + 1];
		uchar v9 = sharedBlock[lid.y + 2][lid.x + 2];
		
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
}

__kernel
__attribute__((reqd_work_group_size(16,16,1)))
void outline4_local(
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);	

	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		uchar v1 = sharedBlock[lid.y    ][lid.x    ];
		uchar v2 = sharedBlock[lid.y    ][lid.x + 1];
		uchar v3 = sharedBlock[lid.y    ][lid.x + 2];
		uchar v4 = sharedBlock[lid.y + 1][lid.x    ];
		uchar v6 = sharedBlock[lid.y + 1][lid.x + 2];
		uchar v7 = sharedBlock[lid.y + 2][lid.x    ];
		uchar v8 = sharedBlock[lid.y + 2][lid.x + 1];
		uchar v9 = sharedBlock[lid.y + 2][lid.x + 2];
		
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
}