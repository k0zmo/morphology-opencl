#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uchar OBJ = 255;
__constant uchar BCK = 0;

__kernel void thinning(
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

__kernel void thinning_priv(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;
	
	uchar v1 = input[(gid.x - 1) + (gid.y - 1) * rowPitch];
	uchar v2 = input[(gid.x    ) + (gid.y - 1) * rowPitch];
	uchar v3 = input[(gid.x + 1) + (gid.y - 1) * rowPitch];
	uchar v4 = input[(gid.x - 1) + (gid.y    ) * rowPitch];
	uchar v6 = input[(gid.x + 1) + (gid.y    ) * rowPitch];
	uchar v7 = input[(gid.x - 1) + (gid.y + 1) * rowPitch];
	uchar v8 = input[(gid.x    ) + (gid.y + 1) * rowPitch];
	uchar v9 = input[(gid.x + 1) + (gid.y + 1) * rowPitch];
	
	if (v1 == OBJ &&
		v2 == OBJ &&
		v3 == OBJ &&
		v4 == OBJ &&
		v6 == OBJ &&
		v7 == OBJ &&
		v8 == OBJ &&
		v9 == OBJ)
	{
		output[gid.x + gid.y * rowPitch] = BCK;
	}
}

__kernel void thinning_priv_lut(
	__global uchar* input,
	__global uchar* output)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	size_t rowPitch = get_global_size(0) + 2;
	
	uchar v1 = input[(gid.x - 1) + (gid.y - 1) * rowPitch];
	uchar v2 = input[(gid.x    ) + (gid.y - 1) * rowPitch];
	uchar v3 = input[(gid.x + 1) + (gid.y - 1) * rowPitch];
	uchar v4 = input[(gid.x - 1) + (gid.y    ) * rowPitch];
	uchar v6 = input[(gid.x + 1) + (gid.y    ) * rowPitch];
	uchar v7 = input[(gid.x - 1) + (gid.y + 1) * rowPitch];
	uchar v8 = input[(gid.x    ) + (gid.y + 1) * rowPitch];
	uchar v9 = input[(gid.x + 1) + (gid.y + 1) * rowPitch];
	
	const uint w = (1 << 8) | (1 << 7) | (1 << 6) | 
		(1 << 5) | (1 << 3) | (1 << 2) | (1 << 1) | 1;
		
	uint v =
		(v1) | 
		(v2 << 1) |
		(v3 << 2) |
		(v4 << 3) |
		(v6 << 5) |
		(v7 << 6) |
		(v8 << 7) |
		(v9 << 8);

	
	if (v == w)
		output[gid.x + gid.y * rowPitch] = BCK;
}

#define WORK_GROUP_SIZE 16
#define SHARED_SIZE WORK_GROUP_SIZE+2

__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE,WORK_GROUP_SIZE,1)))
void thinning_local(
	__global uchar* input,
	__global uchar* output,
	const int2 imageSize)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	int2 localSize = (int2)(WORK_GROUP_SIZE, WORK_GROUP_SIZE);
	int2 groupId = (int2)(get_group_id(0), get_group_id(1));
	int2 groupStartId = groupId * localSize; // id pierwszego bajtu w tej grupie roboczej
	
	//__local uint sharedBlock[sharedSize.x * sharedSize.y];
	__local uchar sharedBlock[SHARED_SIZE * SHARED_SIZE];

	// Zaladuj obszar roboczy obrazu zrodlowego do pamieci lokalnej
	for(int y = lid.y; y < SHARED_SIZE; y += localSize.y)
	{
		int r = groupStartId.y + y; // indeks.y bajtu z wejsca
		for(int x = lid.x; x < SHARED_SIZE; x += localSize.x)
		{
			int c = groupStartId.x + x; // indeks.x bajtu z wejscia
			
			if(c < imageSize.x && r < imageSize.y)
			{
				sharedBlock[x + y * SHARED_SIZE] = input[c + r * imageSize.x];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	uchar v1 = sharedBlock[(lid.x    ) + (lid.y    ) * SHARED_SIZE];
	uchar v2 = sharedBlock[(lid.x + 1) + (lid.y    ) * SHARED_SIZE];
	uchar v3 = sharedBlock[(lid.x + 2) + (lid.y    ) * SHARED_SIZE];
	uchar v4 = sharedBlock[(lid.x    ) + (lid.y + 1) * SHARED_SIZE];
	uchar v6 = sharedBlock[(lid.x + 2) + (lid.y + 1) * SHARED_SIZE];
	uchar v7 = sharedBlock[(lid.x    ) + (lid.y + 2) * SHARED_SIZE];
	uchar v8 = sharedBlock[(lid.x + 1) + (lid.y + 2) * SHARED_SIZE];
	uchar v9 = sharedBlock[(lid.x + 2) + (lid.y + 2) * SHARED_SIZE];
	
	if (v1 == OBJ &&
		v2 == OBJ &&
		v3 == OBJ &&
		v4 == OBJ &&
		v6 == OBJ &&
		v7 == OBJ &&
		v8 == OBJ &&
		v9 == OBJ)
	{
		output[gid.x + gid.y * imageSize.x] = BCK;
	}
}

__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE,WORK_GROUP_SIZE,1)))
void thinning_local_lut(
	__global uchar* input,
	__global uchar* output,
	const int2 imageSize)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	int2 localSize = (int2)(WORK_GROUP_SIZE, WORK_GROUP_SIZE);
	int2 groupId = (int2)(get_group_id(0), get_group_id(1));
	int2 groupStartId = groupId * localSize; // id pierwszego bajtu w tej grupie roboczej
	
	//__local uint sharedBlock[sharedSize.x * sharedSize.y];
	__local uchar sharedBlock[SHARED_SIZE * SHARED_SIZE];

	// Zaladuj obszar roboczy obrazu zrodlowego do pamieci lokalnej
	for(int y = lid.y; y < SHARED_SIZE; y += localSize.y)
	{
		int r = groupStartId.y + y; // indeks.y bajtu z wejsca
		for(int x = lid.x; x < SHARED_SIZE; x += localSize.x)
		{
			int c = groupStartId.x + x; // indeks.x bajtu z wejscia
			
			if(c < imageSize.x && r < imageSize.y)
			{
				sharedBlock[x + y * SHARED_SIZE] = input[c + r * imageSize.x];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	uchar v1 = sharedBlock[(lid.x    ) + (lid.y    ) * SHARED_SIZE];
	uchar v2 = sharedBlock[(lid.x + 1) + (lid.y    ) * SHARED_SIZE];
	uchar v3 = sharedBlock[(lid.x + 2) + (lid.y    ) * SHARED_SIZE];
	uchar v4 = sharedBlock[(lid.x    ) + (lid.y + 1) * SHARED_SIZE];
	uchar v6 = sharedBlock[(lid.x + 2) + (lid.y + 1) * SHARED_SIZE];
	uchar v7 = sharedBlock[(lid.x    ) + (lid.y + 2) * SHARED_SIZE];
	uchar v8 = sharedBlock[(lid.x + 1) + (lid.y + 2) * SHARED_SIZE];
	uchar v9 = sharedBlock[(lid.x + 2) + (lid.y + 2) * SHARED_SIZE];
	
	const uint w = (1 << 8) | (1 << 7) | (1 << 6) | 
		(1 << 5) | (1 << 3) | (1 << 2) | (1 << 1) | 1;
		
	uint v =
		(v1) | 
		(v2 << 1) |
		(v3 << 2) |
		(v4 << 3) |
		(v6 << 5) |
		(v7 << 6) |
		(v8 << 7) |
		(v9 << 8);
	
	if (v == w)
		output[gid.x + gid.y * imageSize.x] = BCK;
}