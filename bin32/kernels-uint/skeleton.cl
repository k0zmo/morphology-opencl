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

#define SHARED_SIZEX 20
#define SHARED_SIZEY 18

__attribute__((always_inline))
void cacheNeighbours(
	__global uint4* input,
	const int2 imageSize,
	__local uint sharedBlock[SHARED_SIZEY][SHARED_SIZEX])
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
	
	__local uint4* sharedBlock4 = (__local uint4*)(&sharedBlock[lid.y][lid.x*4]);
	
	if (gid.y < imageSize.y && 
		gid.x < imageSize.x/4 && 
		lid.y < SHARED_SIZEY)
	{
		sharedBlock4[0] = input[gid.x + gid.y*imageSize.x/4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

#ifdef DDD
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void TEMPLATE_SKELETON_ITER(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy
	// 1|1|1
	// X|1|X
	// 0|0|0
	//
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
	uint v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v1 == OBJ &&
		v2 == OBJ &&
		v3 == OBJ &&
		v4 == OBJ &&
		v5 == OBJ &&
		v6 == OBJ &&
		v7 == OBJ &&
		v8 == OBJ &&
		v9 == OBJ)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		atomic_inc(counter);
	}
}
#endif

__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter1_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy
	// 1|1|1
	// X|1|X
	// 0|0|0
	//
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
	//uint v4 = sharedBlock[lid.y + 1][lid.x    ];
	uint v5 = sharedBlock[lid.y + 1][lid.x + 1];
	//uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v1 == OBJ &&
		v2 == OBJ &&
		v3 == OBJ &&
		v5 == OBJ &&
		v7 == BCK &&
		v8 == BCK &&
		v9 == BCK)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		atomic_inc(counter);
	}
}

__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter2_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy - 90 w lewo
	// 1|X|0
	// 1|1|0
	// 1|x|0
	//
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
	//uint v2 = sharedBlock[lid.y    ][lid.x + 1];
	uint v3 = sharedBlock[lid.y    ][lid.x + 2];
	uint v4 = sharedBlock[lid.y + 1][lid.x    ];
	uint v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	//uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v1 == OBJ &&
		v3 == BCK &&
		v4 == OBJ &&
		v5 == OBJ &&
		v6 == BCK &&
		v7 == OBJ &&
		v9 == BCK)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		atomic_inc(counter);
	}
}

__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter3_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy - 180 w lewo
	// 0|0|0
	// X|1|X
	// 1|1|1
	//
	
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
	//uint v4 = sharedBlock[lid.y + 1][lid.x    ];
	uint v5 = sharedBlock[lid.y + 1][lid.x + 1];
	//uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v1 == BCK &&
		v2 == BCK &&
		v3 == BCK &&
		v5 == OBJ &&
		v7 == OBJ &&
		v8 == OBJ &&
		v9 == OBJ)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		atomic_inc(counter);
	}
}

__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter4_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy - 270 w lewo
	// 0|X|1
	// 0|1|1
	// 0|X|1
	//
	
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
	//uint v2 = sharedBlock[lid.y    ][lid.x + 1];
	uint v3 = sharedBlock[lid.y    ][lid.x + 2];
	uint v4 = sharedBlock[lid.y + 1][lid.x    ];
	uint v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	//uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v1 == BCK &&
		v3 == OBJ &&
		v4 == BCK &&
		v5 == OBJ &&
		v6 == OBJ &&
		v7 == BCK &&
		v9 == OBJ)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		atomic_inc(counter);
	}
}

__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter5_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi
	// X|1|X
	// 0|1|1
	// 0|0|X
	//
	
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
	//uint v1 = sharedBlock[lid.y    ][lid.x    ];
	uint v2 = sharedBlock[lid.y    ][lid.x + 1];
	//uint v3 = sharedBlock[lid.y    ][lid.x + 2];
	uint v4 = sharedBlock[lid.y + 1][lid.x    ];
	uint v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	//uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v2 == OBJ &&
		v4 == BCK &&
		v5 == OBJ &&
		v6 == OBJ &&
		v7 == BCK &&
		v8 == BCK)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		atomic_inc(counter);
	}
}

__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter6_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi - 90 stopni w lewo
	// X|1|X
	// 1|1|0
	// X|0|0
	//
	
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
	//uint v1 = sharedBlock[lid.y    ][lid.x    ];
	uint v2 = sharedBlock[lid.y    ][lid.x + 1];
	//uint v3 = sharedBlock[lid.y    ][lid.x + 2];
	uint v4 = sharedBlock[lid.y + 1][lid.x    ];
	uint v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	//uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v2 == OBJ &&
		v4 == OBJ &&
		v5 == OBJ &&
		v6 == BCK &&
		v8 == BCK &&
		v9 == BCK)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		atomic_inc(counter);
	}
}

__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter7_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi - 180 stopni w lewo
	// X|0|0
	// 1|1|0
	// X|1|X
	//
	
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
	//uint v1 = sharedBlock[lid.y    ][lid.x    ];
	uint v2 = sharedBlock[lid.y    ][lid.x + 1];
	uint v3 = sharedBlock[lid.y    ][lid.x + 2];
	uint v4 = sharedBlock[lid.y + 1][lid.x    ];
	uint v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	//uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	//uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v2 == BCK &&
		v3 == BCK &&
		v4 == OBJ &&
		v5 == OBJ &&
		v6 == BCK &&
		v8 == OBJ)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		atomic_inc(counter);
	}
}

__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter8_local(
	__global uint4* input,
	__global uint* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi - 270 stopni w lewo
	// 0|0|X
	// 0|1|1
	// X|1|X
	//
	
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
	//uint v3 = sharedBlock[lid.y    ][lid.x + 2];
	uint v4 = sharedBlock[lid.y + 1][lid.x    ];
	uint v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uint v6 = sharedBlock[lid.y + 1][lid.x + 2];
	//uint v7 = sharedBlock[lid.y + 2][lid.x    ];
	uint v8 = sharedBlock[lid.y + 2][lid.x + 1];
	//uint v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
	if (v1 == BCK &&
		v2 == BCK &&
		v4 == BCK &&
		v5 == OBJ &&
		v6 == OBJ &&
		v8 == OBJ)
	{
		output[(gid.y+1)*imageSize.x + (gid.x+1)] = BCK;
		atomic_inc(counter);
	}
}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// Naive versions
	
__kernel void skeleton_iter1(
	__global uint* input,
	__global uint* output,
	const int rowPitch,
	counter_type counter)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	// Element strukturalny pierwszy
	// 1|1|1
	// X|1|X
	// 0|0|0
	//
		
	if (input[(gid.x - 1) + (gid.y - 1) * rowPitch] == OBJ &&
		input[(gid.x    ) + (gid.y - 1) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y - 1) * rowPitch] == OBJ &&
		
		input[(gid.x    ) + (gid.y    ) * rowPitch] == OBJ &&
		
		input[(gid.x - 1) + (gid.y + 1) * rowPitch] == BCK &&
		input[(gid.x    ) + (gid.y + 1) * rowPitch] == BCK &&
		input[(gid.x + 1) + (gid.y + 1) * rowPitch] == BCK)
	{
		output[gid.x + gid.y * rowPitch] = BCK;
		atomic_inc(counter);
	}
}

__kernel void skeleton_iter2(
	__global uint* input,
	__global uint* output,
	const int rowPitch,
	counter_type counter)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
		
	// Element strukturalny pierwszy - 90 w lewo
	// 1|X|0
	// 1|1|0
	// 1|x|0
	//
	
	if (input[(gid.x - 1) + (gid.y - 1) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y - 1) * rowPitch] == BCK &&
		
		input[(gid.x - 1) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x    ) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * rowPitch] == BCK &&
		
		input[(gid.x - 1) + (gid.y + 1) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y + 1) * rowPitch] == BCK)
	{
		output[gid.x + gid.y * rowPitch] = BCK;
		atomic_inc(counter);
	}
}

__kernel void skeleton_iter3(
	__global uint* input,
	__global uint* output,
	const int rowPitch,
	counter_type counter)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));

	// Element strukturalny pierwszy - 180 w lewo
	// 0|0|0
	// X|1|X
	// 1|1|1
	//
		
	if (input[(gid.x - 1) + (gid.y - 1) * rowPitch] == BCK &&
		input[(gid.x    ) + (gid.y - 1) * rowPitch] == BCK &&
		input[(gid.x + 1) + (gid.y - 1) * rowPitch] == BCK &&
		input[(gid.x    ) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x - 1) + (gid.y + 1) * rowPitch] == OBJ &&
		input[(gid.x    ) + (gid.y + 1) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y + 1) * rowPitch] == OBJ)
	{
		output[gid.x + gid.y * rowPitch] = BCK;
		atomic_inc(counter);
	}
}

__kernel void skeleton_iter4(
	__global uint* input,
	__global uint* output,
	const int rowPitch,
	counter_type counter)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	// Element strukturalny pierwszy - 270 w lewo
	// 0|X|1
	// 0|1|1
	// 0|X|1
	//
	
	if (input[(gid.x - 1) + (gid.y - 1) * rowPitch] == BCK &&
		input[(gid.x + 1) + (gid.y - 1) * rowPitch] == OBJ &&
		input[(gid.x - 1) + (gid.y    ) * rowPitch] == BCK &&
		input[(gid.x    ) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x - 1) + (gid.y + 1) * rowPitch] == BCK &&
		input[(gid.x + 1) + (gid.y + 1) * rowPitch] == OBJ)
	{
		output[gid.x + gid.y * rowPitch] = BCK;
		atomic_inc(counter);
	}
}

__kernel void skeleton_iter5(
	__global uint* input,
	__global uint* output,
	const int rowPitch,
	counter_type counter)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));

	// Element strukturalny drugi
	// X|1|X
	// 0|1|1
	// 0|0|X
	//

	if (input[(gid.x    ) + (gid.y - 1) * rowPitch] == OBJ &&
		input[(gid.x - 1) + (gid.y    ) * rowPitch] == BCK &&
		input[(gid.x    ) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x - 1) + (gid.y + 1) * rowPitch] == BCK &&
		input[(gid.x    ) + (gid.y + 1) * rowPitch] == BCK)
	{
		output[gid.x + gid.y * rowPitch] = BCK;
		atomic_inc(counter);
	}
}

__kernel void skeleton_iter6(
	__global uint* input,
	__global uint* output,
	const int rowPitch,
	counter_type counter)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
		
	// Element strukturalny drugi - 90 stopni w lewo
	// X|1|X
	// 1|1|0
	// X|0|0
	//
		
	if (input[(gid.x    ) + (gid.y - 1) * rowPitch] == OBJ &&
		input[(gid.x - 1) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x    ) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * rowPitch] == BCK &&
		input[(gid.x    ) + (gid.y + 1) * rowPitch] == BCK &&
		input[(gid.x + 1) + (gid.y + 1) * rowPitch] == BCK)
	{
		output[gid.x + gid.y * rowPitch] = BCK;
		atomic_inc(counter);
	}
}

__kernel void skeleton_iter7(
	__global uint* input,
	__global uint* output,
	const int rowPitch,
	counter_type counter)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
		
	// Element strukturalny drugi - 180 stopni w lewo
	// X|0|0
	// 1|1|0
	// X|1|X
	//
		
	if (input[(gid.x    ) + (gid.y - 1) * rowPitch] == BCK &&
		input[(gid.x + 1) + (gid.y - 1) * rowPitch] == BCK &&
		input[(gid.x - 1) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x    ) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * rowPitch] == BCK &&
		input[(gid.x    ) + (gid.y + 1) * rowPitch] == OBJ)
	{
		output[gid.x + gid.y * rowPitch] = BCK;
		atomic_inc(counter);
	}
}

__kernel void skeleton_iter8(
	__global uint* input,
	__global uint* output,
	const int rowPitch,
	counter_type counter)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
		
	// Element strukturalny drugi - 270 stopni w lewo
	// 0|0|X
	// 0|1|1
	// X|1|X
	//
		
	if (input[(gid.x - 1) + (gid.y - 1) * rowPitch] == BCK &&
		input[(gid.x    ) + (gid.y - 1) * rowPitch] == BCK &&
		input[(gid.x - 1) + (gid.y    ) * rowPitch] == BCK &&
		input[(gid.x    ) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x + 1) + (gid.y    ) * rowPitch] == OBJ &&
		input[(gid.x    ) + (gid.y + 1) * rowPitch] == OBJ)
	{
		output[gid.x + gid.y * rowPitch] = BCK;
		atomic_inc(counter);
	}
}