#include "common.cl"

__kernel void outline(
	__global type_t* input,
	__global type_t* output,
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
	__global type4_t* input,
	__global type_t* output,
	const int2 imageSize)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	__local type_t sharedBlock[SHARED_SIZEY*SHARED_SIZEX];
	cache4ToLocalMemory16(input, imageSize, lid, sharedBlock);	

	if (gid.y < imageSize.y - 2 &&
		gid.x < imageSize.x - 2)
	{
		type_t v1 = sharedBlock[mad24(SHARED_SIZEX, (lid.y    ), lid.x    )];
		type_t v2 = sharedBlock[mad24(SHARED_SIZEX, (lid.y    ), lid.x + 1)];
		type_t v3 = sharedBlock[mad24(SHARED_SIZEX, (lid.y    ), lid.x + 2)];
		type_t v4 = sharedBlock[mad24(SHARED_SIZEX, (lid.y + 1), lid.x    )];
		type_t v6 = sharedBlock[mad24(SHARED_SIZEX, (lid.y + 1), lid.x + 2)];
		type_t v7 = sharedBlock[mad24(SHARED_SIZEX, (lid.y + 2), lid.x    )];
		type_t v8 = sharedBlock[mad24(SHARED_SIZEX, (lid.y + 2), lid.x + 1)];
		type_t v9 = sharedBlock[mad24(SHARED_SIZEX, (lid.y + 2), lid.x + 2)];
		
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