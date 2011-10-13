#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uchar OBJ = 255;
__constant uchar BCK = 0;

__kernel void skeleton_iter1(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;
	
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
	}
}

__kernel void skeleton_iter2(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;
		
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
	}
}

__kernel void skeleton_iter3(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;

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
	}
}

__kernel void skeleton_iter4(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;

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
	}
}

__kernel void skeleton_iter5(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;

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
	}
}

__kernel void skeleton_iter6(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;
		
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
	}
}

__kernel void skeleton_iter7(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;
		
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
	}
}

__kernel void skeleton_iter8(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;
		
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
	}
}