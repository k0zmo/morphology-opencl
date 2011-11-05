#include "common.cl"

__kernel void outline(
	__global uchar* input,
	__global uchar* output,
	const int2 imageSize)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		if (input[(gid.x - 1) + (gid.y - 1) * imageSize.x] == OBJ &&
			input[(gid.x    ) + (gid.y - 1) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y - 1) * imageSize.x] == OBJ &&
			
			input[(gid.x - 1) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y    ) * imageSize.x] == OBJ &&
			
			input[(gid.x - 1) + (gid.y + 1) * imageSize.x] == OBJ &&
			input[(gid.x    ) + (gid.y + 1) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y + 1) * imageSize.x] == OBJ)
		{
			output[gid.x + gid.y * imageSize.x] = BCK;
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