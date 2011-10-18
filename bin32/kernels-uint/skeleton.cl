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