// Dla adresowania uint8*
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uchar erodeINF = 255;
__constant uchar dilateINF = 0;
__constant uchar OBJ = 255;
__constant uchar BCK = 0;

__kernel void subtract(
	__global uchar* a,
	__global uchar* b,
	__global uchar* output)
{
	size_t gid = get_global_id(0);
	output[gid] = (b[gid] > a[gid]) ? (0): (a[gid] - b[gid]);
}

__kernel void addHalf(
	__global uchar* skeleton,
	__global uchar* src)
{
	size_t gid = get_global_id(0);
	
	if(skeleton[gid] == 0)
		skeleton[gid] = src[gid] / 2;
}

__kernel void erode(
	__global uchar* input,
	__global uchar* output,
	__constant int2* coords,
	const int coords_size,
	const int rowPitch)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	uchar val = erodeINF;
	
	for(int i = 0; i < coords_size; ++i)
	{
		int2 coord = coords[i] + gid;
		val = min(val, input[coord.x + coord.y * rowPitch]);
	}

	output[gid.x + gid.y * rowPitch] = val;
}

__kernel void dilate(
	__global uchar* input,
	__global uchar* output,
	__constant int2* coords,
	const int coords_size,
	const int rowPitch)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	uchar val = dilateINF;
	
	for(int i = 0; i < coords_size; ++i)
	{
		int2 coord = coords[i] + gid;
		val = max(val, input[coord.x + coord.y * rowPitch]);
	}

	output[gid.x + gid.y * rowPitch] = val;
}

__kernel void remove(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;
	
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