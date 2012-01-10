#include "common.cl"

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter1_local(
	__global type4_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy
	// 1|1|1
	// X|1|X
	// 0|0|0
	//
	
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local type_t sharedBlock[SHARED_SIZEY*SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);

	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		type_t v1 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x    )];
		type_t v2 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 1)];
		type_t v3 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 2)];
		type_t v5 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 1)];
		type_t v7 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x    )];
		type_t v8 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 1)];
		type_t v9 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 2)];
		
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
}

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter2_local(
	__global type4_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy - 90 w lewo
	// 1|X|0
	// 1|1|0
	// 1|x|0
	//
	
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local type_t sharedBlock[SHARED_SIZEY*SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		type_t v1 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x    )];
		type_t v3 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 2)];
		type_t v4 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x    )];
		type_t v5 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 1)];
		type_t v6 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 2)];
		type_t v7 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x    )];
		type_t v9 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 2)];
		
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
}

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter3_local(
	__global type4_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy - 180 w lewo
	// 0|0|0
	// X|1|X
	// 1|1|1
	//
	
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local type_t sharedBlock[SHARED_SIZEY*SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		type_t v1 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x    )];
		type_t v2 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 1)];
		type_t v3 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 2)];
		type_t v5 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 1)];
		type_t v7 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x    )];
		type_t v8 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 1)];
		type_t v9 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 2)];
		
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
}

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter4_local(
	__global type4_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny pierwszy - 270 w lewo
	// 0|X|1
	// 0|1|1
	// 0|X|1
	//
	
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local type_t sharedBlock[SHARED_SIZEY*SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		type_t v1 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x    )];
		type_t v3 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 2)];
		type_t v4 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x    )];
		type_t v5 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 1)];
		type_t v6 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 2)];
		type_t v7 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x    )];
		type_t v9 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 2)];
		
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
}

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter5_local(
	__global type4_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi
	// X|1|X
	// 0|1|1
	// 0|0|X
	//
	
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local type_t sharedBlock[SHARED_SIZEY*SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		type_t v2 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 1)];
		type_t v4 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x    )];
		type_t v5 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 1)];
		type_t v6 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 2)];
		type_t v7 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x    )];
		type_t v8 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 1)];

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
}

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter6_local(
	__global type4_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi - 90 stopni w lewo
	// X|1|X
	// 1|1|0
	// X|0|0
	//
	
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local type_t sharedBlock[SHARED_SIZEY*SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		type_t v2 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 1)];
		type_t v4 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x    )];
		type_t v5 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 1)];
		type_t v6 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 2)];
		type_t v8 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 1)];
		type_t v9 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 2)];
		
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
}

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter7_local(
	__global type4_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi - 180 stopni w lewo
	// X|0|0
	// 1|1|0
	// X|1|X
	//
	
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local type_t sharedBlock[SHARED_SIZEY*SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);
	
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		type_t v2 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 1)];
		type_t v3 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 2)];
		type_t v4 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x    )];
		type_t v5 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 1)];
		type_t v6 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 2)];
		type_t v8 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 1)];
		
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
}

__kernel
__attribute__((reqd_work_group_size(16, 16, 1)))
void skeleton4_iter8_local(
	__global type4_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	// Element strukturalny drugi - 270 stopni w lewo
	// 0|0|X
	// 0|1|1
	// X|1|X
	//
	
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local type_t sharedBlock[SHARED_SIZEY*SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);

	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		type_t v1 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x    )];
		type_t v2 = sharedBlock[mad24(SHARED_SIZEX, lid.y    , lid.x + 1)];
		type_t v4 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x    )];
		type_t v5 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 1)];
		type_t v6 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 1, lid.x + 2)];
		type_t v8 = sharedBlock[mad24(SHARED_SIZEX, lid.y + 2, lid.x + 1)];
		
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
}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// Naive versions
	
__kernel void skeleton_iter1(
	__global type_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	
	// Element strukturalny pierwszy
	// 1|1|1
	// X|1|X
	// 0|0|0
	//
	
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		if (input[(gid.x - 1) + (gid.y - 1) * imageSize.x] == OBJ &&
			input[(gid.x    ) + (gid.y - 1) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y - 1) * imageSize.x] == OBJ &&
			
			input[(gid.x    ) + (gid.y    ) * imageSize.x] == OBJ &&
			
			input[(gid.x - 1) + (gid.y + 1) * imageSize.x] == BCK &&
			input[(gid.x    ) + (gid.y + 1) * imageSize.x] == BCK &&
			input[(gid.x + 1) + (gid.y + 1) * imageSize.x] == BCK)
		{
			output[gid.x + gid.y * imageSize.x] = BCK;
			atomic_inc(counter);
		}	
	}
}

__kernel void skeleton_iter2(
	__global type_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
		
	// Element strukturalny pierwszy - 90 w lewo
	// 1|X|0
	// 1|1|0
	// 1|x|0
	//
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{	
		if (input[(gid.x - 1) + (gid.y - 1) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y - 1) * imageSize.x] == BCK &&
			
			input[(gid.x - 1) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x    ) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y    ) * imageSize.x] == BCK &&
			
			input[(gid.x - 1) + (gid.y + 1) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y + 1) * imageSize.x] == BCK)
		{
			output[gid.x + gid.y * imageSize.x] = BCK;
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter3(
	__global type_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };

	// Element strukturalny pierwszy - 180 w lewo
	// 0|0|0
	// X|1|X
	// 1|1|1
	//
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		if (input[(gid.x - 1) + (gid.y - 1) * imageSize.x] == BCK &&
			input[(gid.x    ) + (gid.y - 1) * imageSize.x] == BCK &&
			input[(gid.x + 1) + (gid.y - 1) * imageSize.x] == BCK &&
			input[(gid.x    ) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x - 1) + (gid.y + 1) * imageSize.x] == OBJ &&
			input[(gid.x    ) + (gid.y + 1) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y + 1) * imageSize.x] == OBJ)
		{
			output[gid.x + gid.y * imageSize.x] = BCK;
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter4(
	__global type_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	
	// Element strukturalny pierwszy - 270 w lewo
	// 0|X|1
	// 0|1|1
	// 0|X|1
	//
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{	
		if (input[(gid.x - 1) + (gid.y - 1) * imageSize.x] == BCK &&
			input[(gid.x + 1) + (gid.y - 1) * imageSize.x] == OBJ &&
			input[(gid.x - 1) + (gid.y    ) * imageSize.x] == BCK &&
			input[(gid.x    ) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x - 1) + (gid.y + 1) * imageSize.x] == BCK &&
			input[(gid.x + 1) + (gid.y + 1) * imageSize.x] == OBJ)
		{
			output[gid.x + gid.y * imageSize.x] = BCK;
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter5(
	__global type_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };

	// Element strukturalny drugi
	// X|1|X
	// 0|1|1
	// 0|0|X
	//
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		if (input[(gid.x    ) + (gid.y - 1) * imageSize.x] == OBJ &&
			input[(gid.x - 1) + (gid.y    ) * imageSize.x] == BCK &&
			input[(gid.x    ) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x - 1) + (gid.y + 1) * imageSize.x] == BCK &&
			input[(gid.x    ) + (gid.y + 1) * imageSize.x] == BCK)
		{
			output[gid.x + gid.y * imageSize.x] = BCK;
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter6(
	__global type_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
		
	// Element strukturalny drugi - 90 stopni w lewo
	// X|1|X
	// 1|1|0
	// X|0|0
	//
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{		
		if (input[(gid.x    ) + (gid.y - 1) * imageSize.x] == OBJ &&
			input[(gid.x - 1) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x    ) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y    ) * imageSize.x] == BCK &&
			input[(gid.x    ) + (gid.y + 1) * imageSize.x] == BCK &&
			input[(gid.x + 1) + (gid.y + 1) * imageSize.x] == BCK)
		{
			output[gid.x + gid.y * imageSize.x] = BCK;
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter7(
	__global type_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
		
	// Element strukturalny drugi - 180 stopni w lewo
	// X|0|0
	// 1|1|0
	// X|1|X
	//
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		if (input[(gid.x    ) + (gid.y - 1) * imageSize.x] == BCK &&
			input[(gid.x + 1) + (gid.y - 1) * imageSize.x] == BCK &&
			input[(gid.x - 1) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x    ) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y    ) * imageSize.x] == BCK &&
			input[(gid.x    ) + (gid.y + 1) * imageSize.x] == OBJ)
		{
			output[gid.x + gid.y * imageSize.x] = BCK;
			atomic_inc(counter);
		}
	}
}

__kernel void skeleton_iter8(
	__global type_t* input,
	__global type_t* output,
	const int2 imageSize,
	counter_type counter)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
		
	// Element strukturalny drugi - 270 stopni w lewo
	// 0|0|X
	// 0|1|1
	// X|1|X
	//
	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{		
		if (input[(gid.x - 1) + (gid.y - 1) * imageSize.x] == BCK &&
			input[(gid.x    ) + (gid.y - 1) * imageSize.x] == BCK &&
			input[(gid.x - 1) + (gid.y    ) * imageSize.x] == BCK &&
			input[(gid.x    ) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x + 1) + (gid.y    ) * imageSize.x] == OBJ &&
			input[(gid.x    ) + (gid.y + 1) * imageSize.x] == OBJ)
		{
			output[gid.x + gid.y * imageSize.x] = BCK;
			atomic_inc(counter);
		}
	}
}