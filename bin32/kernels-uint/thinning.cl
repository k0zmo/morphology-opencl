__constant uint OBJ = 255;
__constant uint BCK = 0;

__kernel void thinning(
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
#define SHARED_SIZE WORK_GROUP_SIZE+2

__kernel __attribute__((reqd_work_group_size(16,16,1)))
void thinning_local(
	__global uint* input,
	__global uint* output,
	const int2 imageSize)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	int2 localSize = (int2)(get_local_size(0), get_local_size(1));
	
	// doesn't work
#if 0
	__local uint sharedBlock[SHARED_SIZE * SHARED_SIZE];
	
	if(gid.x < imageSize.x && gid.y < imageSize.y)
	{
		// Wczytaj piksel do pamieci lokalnej odpowiadajacy temu w pamieci globalnej
		sharedBlock[lid.y*SHARED_SIZE + lid.x] = input[gid.y*imageSize.x + gid.x];
		
		// Wczytaj piksele z 'apron'
		if(lid.x < 2)
		{
			int x = gid.x + localSize.x ;
			if(x < imageSize.x)
				sharedBlock[lid.y*SHARED_SIZE + lid.x + localSize.x] = input[gid.y*imageSize.x + x];
		}
		
		if(lid.y < 2)
		{
			int y = gid.y + localSize.y;
			if(y < imageSize.y)
				sharedBlock[(lid.y + localSize.y)*SHARED_SIZE + lid.x] = input[y*imageSize.x + gid.x];
		}
		
		if(lid.x > 13 && lid.y > 13)
		{
			int x = gid.x + 2;
			int y = gid.y + 2;
			
			if(x < imageSize.x && y < imageSize.y)
				sharedBlock[(lid.y + 2)*SHARED_SIZE + lid.x + 2] = input[y*imageSize.x + x];
		}
	}		
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;
		
	if(lid.x == localSize.x - 1 || lid.y == localSize.y - 1)
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = 128;
	else
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = sharedBlock[(lid.y + 1)*SHARED_SIZE + lid.x + 1];
	return;
	
	uint v1 = sharedBlock[(lid.y    )*SHARED_SIZE + lid.x    ];
	uint v2 = sharedBlock[(lid.y    )*SHARED_SIZE + lid.x + 1];
	uint v3 = sharedBlock[(lid.y    )*SHARED_SIZE + lid.x + 2];
	uint v4 = sharedBlock[(lid.y + 1)*SHARED_SIZE + lid.x    ];
	uint v6 = sharedBlock[(lid.y + 1)*SHARED_SIZE + lid.x + 2];
	uint v7 = sharedBlock[(lid.y + 2)*SHARED_SIZE + lid.x    ];
	uint v8 = sharedBlock[(lid.y + 2)*SHARED_SIZE + lid.x + 1];
	uint v9 = sharedBlock[(lid.y + 2)*SHARED_SIZE + lid.x + 2];
	
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
#else
	__local uint sharedBlock[SHARED_SIZE][SHARED_SIZE];
	
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
#endif
}

#undef SHARED_SIZE
#define SHARED_SIZEX 20
#define SHARED_SIZEY 18

__kernel __attribute__((reqd_work_group_size(16,16,1)))
void thinning4_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize)
{
	int2 localSize = (int2)(get_local_size(0), get_local_size(1));
	int2 groupId = (int2)(get_group_id(0), get_group_id(1));
	int2 groupStartId = groupId * localSize; // id pierwszego bajtu w tej grupie roboczej
	
	// Przebiega od 0 do 255
	int flatLid = get_local_id(0) + get_local_id(1) * localSize.x;
	
	int2 lid;
	lid.x = (flatLid % (SHARED_SIZEX/4));
	lid.y = (flatLid / (SHARED_SIZEX/4));	
	
	int2 gid;
	gid.x = groupStartId.x/4 + lid.x;
	gid.y = groupStartId.y   + lid.y;	
	
	__local uint sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	__local uint4* sharedBlock4 = (__local uint4*)(&sharedBlock[lid.y][lid.x*4]);
	
	if (gid.y < imageSize.y && 
		gid.x < imageSize.x/4 && 
		lid.y < SHARED_SIZEY)
	{
		sharedBlock4[0] = input[gid.x + gid.y*imageSize.x/4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	gid = (int2)(get_global_id(0), get_global_id(1));
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;
		
	lid = (int2)(get_local_id(0), get_local_id(1));	
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