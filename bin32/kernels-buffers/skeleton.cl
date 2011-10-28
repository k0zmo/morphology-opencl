#include "common.cl"

#ifdef USE_ATOMIC_COUNTERS
#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable 
#define counter_type counter32_t
#else
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define counter_type __global uint*
#define atomic_inc atom_inc
#endif

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter1_local(
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy
	// 1|1|1
	// X|1|X
	// 0|0|0
	//
	
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);

	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;		
	
	uchar v1 = sharedBlock[lid.y    ][lid.x    ];
	uchar v2 = sharedBlock[lid.y    ][lid.x + 1];
	uchar v3 = sharedBlock[lid.y    ][lid.x + 2];
	//uchar v4 = sharedBlock[lid.y + 1][lid.x    ];
	uchar v5 = sharedBlock[lid.y + 1][lid.x + 1];
	//uchar v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uchar v7 = sharedBlock[lid.y + 2][lid.x    ];
	uchar v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uchar v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
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

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter2_local(
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy - 90 w lewo
	// 1|X|0
	// 1|1|0
	// 1|x|0
	//
	
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;
		
	uchar v1 = sharedBlock[lid.y    ][lid.x    ];
	//uchar v2 = sharedBlock[lid.y    ][lid.x + 1];
	uchar v3 = sharedBlock[lid.y    ][lid.x + 2];
	uchar v4 = sharedBlock[lid.y + 1][lid.x    ];
	uchar v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uchar v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uchar v7 = sharedBlock[lid.y + 2][lid.x    ];
	//uchar v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uchar v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
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

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter3_local(
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy - 180 w lewo
	// 0|0|0
	// X|1|X
	// 1|1|1
	//
	
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;

	uchar v1 = sharedBlock[lid.y    ][lid.x    ];
	uchar v2 = sharedBlock[lid.y    ][lid.x + 1];
	uchar v3 = sharedBlock[lid.y    ][lid.x + 2];
	//uchar v4 = sharedBlock[lid.y + 1][lid.x    ];
	uchar v5 = sharedBlock[lid.y + 1][lid.x + 1];
	//uchar v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uchar v7 = sharedBlock[lid.y + 2][lid.x    ];
	uchar v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uchar v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
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

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter4_local(
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy - 270 w lewo
	// 0|X|1
	// 0|1|1
	// 0|X|1
	//
	
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;

	uchar v1 = sharedBlock[lid.y    ][lid.x    ];
	//uchar v2 = sharedBlock[lid.y    ][lid.x + 1];
	uchar v3 = sharedBlock[lid.y    ][lid.x + 2];
	uchar v4 = sharedBlock[lid.y + 1][lid.x    ];
	uchar v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uchar v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uchar v7 = sharedBlock[lid.y + 2][lid.x    ];
	//uchar v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uchar v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
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

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter5_local(
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi
	// X|1|X
	// 0|1|1
	// 0|0|X
	//
	
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;

	//uchar v1 = sharedBlock[lid.y    ][lid.x    ];
	uchar v2 = sharedBlock[lid.y    ][lid.x + 1];
	//uchar v3 = sharedBlock[lid.y    ][lid.x + 2];
	uchar v4 = sharedBlock[lid.y + 1][lid.x    ];
	uchar v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uchar v6 = sharedBlock[lid.y + 1][lid.x + 2];
	uchar v7 = sharedBlock[lid.y + 2][lid.x    ];
	uchar v8 = sharedBlock[lid.y + 2][lid.x + 1];
	//uchar v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
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

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter6_local(
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi - 90 stopni w lewo
	// X|1|X
	// 1|1|0
	// X|0|0
	//
	
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;
		
	//uchar v1 = sharedBlock[lid.y    ][lid.x    ];
	uchar v2 = sharedBlock[lid.y    ][lid.x + 1];
	//uchar v3 = sharedBlock[lid.y    ][lid.x + 2];
	uchar v4 = sharedBlock[lid.y + 1][lid.x    ];
	uchar v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uchar v6 = sharedBlock[lid.y + 1][lid.x + 2];
	//uchar v7 = sharedBlock[lid.y + 2][lid.x    ];
	uchar v8 = sharedBlock[lid.y + 2][lid.x + 1];
	uchar v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
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

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter7_local(
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi - 180 stopni w lewo
	// X|0|0
	// 1|1|0
	// X|1|X
	//
	
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;

	//uchar v1 = sharedBlock[lid.y    ][lid.x    ];
	uchar v2 = sharedBlock[lid.y    ][lid.x + 1];
	uchar v3 = sharedBlock[lid.y    ][lid.x + 2];
	uchar v4 = sharedBlock[lid.y + 1][lid.x    ];
	uchar v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uchar v6 = sharedBlock[lid.y + 1][lid.x + 2];
	//uchar v7 = sharedBlock[lid.y + 2][lid.x    ];
	uchar v8 = sharedBlock[lid.y + 2][lid.x + 1];
	//uchar v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
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

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter8_local(
	__global uchar4* input,
	__global uchar* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi - 270 stopni w lewo
	// 0|0|X
	// 0|1|1
	// X|1|X
	//
	
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	
	__local uchar sharedBlock[SHARED_SIZEY][SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - 2)
		return;
		
	if(gid.x >= imageSize.x - 2)
		return;
		
	uchar v1 = sharedBlock[lid.y    ][lid.x    ];
	uchar v2 = sharedBlock[lid.y    ][lid.x + 1];
	//uchar v3 = sharedBlock[lid.y    ][lid.x + 2];
	uchar v4 = sharedBlock[lid.y + 1][lid.x    ];
	uchar v5 = sharedBlock[lid.y + 1][lid.x + 1];
	uchar v6 = sharedBlock[lid.y + 1][lid.x + 2];
	//uchar v7 = sharedBlock[lid.y + 2][lid.x    ];
	uchar v8 = sharedBlock[lid.y + 2][lid.x + 1];
	//uchar v9 = sharedBlock[lid.y + 2][lid.x + 2];
	
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
	__global uchar* input,
	__global uchar* output,
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
	__global uchar* input,
	__global uchar* output,
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
	__global uchar* input,
	__global uchar* output,
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
	__global uchar* input,
	__global uchar* output,
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
	__global uchar* input,
	__global uchar* output,
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
	__global uchar* input,
	__global uchar* output,
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
	__global uchar* input,
	__global uchar* output,
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
	__global uchar* input,
	__global uchar* output,
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