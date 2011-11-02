#include "common.cl"

__kernel void erode(
	__global uint* input,
	__global uint* output,
	__constant int2* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	uint val = erodeINF;
	
	for(int i = 0; i < seSize.z; ++i)
	{
		int2 coord = coords[i] + gid;
		val = min(val, input[coord.x + coord.y * imageSize.x]);
	}
	
	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;
}

__kernel void erode_c4(
	__global uint* input,
	__global uint* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	uint val = erodeINF;
	int c2 = seSize.z >> 1;
	
	for(int i = 0; i < c2; ++i)
	{
		int4 coord = coords[i] + (int4)(gid, gid);
		val = min(val, input[coord.x + coord.y * imageSize.x]);
		val = min(val, input[coord.z + coord.w * imageSize.x]);
	}
	
	if(seSize.z % 2)
	{
		__constant int2* c = (__constant int2*)(coords);
		int2 coord = c[seSize.z-1] + gid;
		val = min(val, input[coord.x + coord.y * imageSize.x]);
	}

	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;
}

__kernel void erode_local(
	__global uint* input,
	__global uint* output,
	__constant int2* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uint* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cacheToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		// Filtracja wlasciwa
		uint val = erodeINF;
		for(int i = 0; i < seSize.z; ++i)
		{
			int2 coord = coords[i] + lid;
			val = min(val, sharedBlock[mad24(coord.y, sharedSize.x, coord.x)]);
		}
		
		output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;	
	}
}

__kernel void erode_c4_local(
	__global uint* input,
	__global uint* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uint* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cacheToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		// Filtracja wlasciwa
		uint val = erodeINF;
		int c2 = seSize.z >> 1;
		
		for(int i = 0; i < c2; ++i)
		{
			int4 coord = coords[i] + (int4)(lid, lid);
			val = min(val, sharedBlock[mad24(coord.y, sharedSize.x, coord.x)]);
			val = min(val, sharedBlock[mad24(coord.w, sharedSize.x, coord.z)]);
		}
		
		if(seSize.z % 2)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[seSize.z-1] + lid;
			val = min(val, sharedBlock[mad24(coord.y, sharedSize.x, coord.x)]);
		}
		
		output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;	
	}
}

__kernel
__attribute__((reqd_work_group_size(16,16,1))) 
void erode4_local(
	__global uint4* input,
	__global uint* output,
	__constant int2* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uint* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cache4ToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		// Filtracja wlasciwa
		uint val = erodeINF;

		for(int i = 0; i < seSize.z; ++i)
		{
			int2 coord = coords[i] + lid;
			val = min(val, sharedBlock[mad24(coord.y, sharedSize.x, coord.x)]);
		}
		
		output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;
	}
}

__kernel
__attribute__((reqd_work_group_size(16,16,1))) 
void erode4_c4_local(
	__global uint4* input,
	__global uint* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uint* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cache4ToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		// Filtracja wlasciwa
		uint val = erodeINF;	
		int c2 = seSize.z >> 1;
		
		for(int i = 0; i < c2; ++i)
		{
			int4 coord = coords[i] + (int4)(lid, lid);
			val = min(val, sharedBlock[mad24(coord.y, sharedSize.x, coord.x)]);
			val = min(val, sharedBlock[mad24(coord.w, sharedSize.x, coord.z)]);
		}
		
		if(seSize.z % 2)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[seSize.z-1] + lid;
			val = min(val, sharedBlock[mad24(coord.y, sharedSize.x, coord.x)]);
		}
		
		output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;	
	}
}

#ifndef COORDS_SIZE
#define COORDS_SIZE 169
#endif

__kernel
__attribute__((work_group_size_hint(16,16,1))) 
void erode4_c4_local_def(
	__global uint4* input,
	__global uint* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uint* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cache4ToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		// Filtracja wlasciwa
		uint val = erodeINF;	
		int c2 = COORDS_SIZE / 2;
		
		#pragma unroll
		for(int i = 0; i < c2; ++i)
		{
			int4 coord = coords[i] + (int4)(lid, lid);
			val = min(val, sharedBlock[mad24(coord.y, sharedSize.x, coord.x)]);
			val = min(val, sharedBlock[mad24(coord.w, sharedSize.x, coord.z)]);
		}
		
		if(COORDS_SIZE % 2)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[COORDS_SIZE-1] + lid;
			val = min(val, sharedBlock[mad24(coord.y, sharedSize.x, coord.x)]);
		}
		
		output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;	
	}
}