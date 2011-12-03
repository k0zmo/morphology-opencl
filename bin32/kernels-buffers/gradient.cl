#include "common.cl"

__kernel void gradient(
	__global uchar* input,
	__global uchar* output,
	__constant int2* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize)
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		uchar minval = erodeINF;
		uchar maxval = dilateINF;
		
		for(int i = 0; i < seSize.z; ++i)
		{
			int2 coord = coords[i] + gid;
			uchar v = input[coord.x + coord.y * imageSize.x];
			
			minval = min(minval, v);
			maxval = max(maxval, v);
		}
		
		output[(gid.x + seSize.x) +
			(gid.y + seSize.y) * imageSize.x] = sub_sat(maxval, minval);	
	}
}

__kernel void gradient_local(
	__global uchar* input,
	__global uchar* output,
	__constant int2* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cacheToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		uchar minval = erodeINF;
		uchar maxval = dilateINF;
		
		for(int i = 0; i < seSize.z; ++i)
		{
			int2 coord = coords[i] + lid;
			uchar v = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			
			minval = min(minval, v);
			maxval = max(maxval, v);
		}
			
		output[(gid.x + seSize.x) + 
			(gid.y + seSize.y) * imageSize.x] = sub_sat(maxval, minval);
	}
}

__kernel void gradient_c4_local(
	__global uchar* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cacheToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		uchar minval = erodeINF;
		uchar maxval = dilateINF;
		int c2 = seSize.z >> 1;
		
		for(int i = 0; i < c2; ++i)
		{
			int4 coord = coords[i] + (int4)(lid, lid);

			uchar v0 = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			uchar v1 = sharedBlock[mad24(coord.w, sharedSize.x, coord.z)];
			
			minval = min(minval, v0);
			minval = min(minval, v1);
			maxval = max(maxval, v0);
			maxval = max(maxval, v1);
		}
		
		if(seSize.z % 2)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[seSize.z-1] + lid;
			
			uchar v = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			
			minval = min(minval, v);
			maxval = max(maxval, v);
		}
		
		output[(gid.x + seSize.x) + 
			(gid.y + seSize.y)* imageSize.x] = sub_sat(maxval, minval);
	}
}

__kernel void gradient_c4_local_unroll(
	__global uchar* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cacheToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		uchar minval = erodeINF;
		uchar maxval = dilateINF;
		int c2 = seSize.z >> 1;
		int i = 0;
		
		for(; i < c2; i += 2)
		{
			int4 coord0 = coords[i]   + (int4)(lid, lid);
			int4 coord1 = coords[i+1] + (int4)(lid, lid);
			
			uchar v0 = sharedBlock[mad24(coord0.y, sharedSize.x, coord0.x)];
			uchar v1 = sharedBlock[mad24(coord0.w, sharedSize.x, coord0.z)];
			uchar v2 = sharedBlock[mad24(coord1.y, sharedSize.x, coord1.x)];
			uchar v3 = sharedBlock[mad24(coord1.w, sharedSize.x, coord1.z)];
			
			minval = min(minval, v0);
			minval = min(minval, v1);
			minval = min(minval, v2);
			minval = min(minval, v3);
			maxval = max(maxval, v0);
			maxval = max(maxval, v1);
			maxval = max(maxval, v2);
			maxval = max(maxval, v3);
		}
		
		i *= 2;
		
		for( ; i < seSize.z; ++i)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[i] + lid;
			
			uchar v = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			
			minval = min(minval, v);
			maxval = max(maxval, v);
		}
		
		output[(gid.x + seSize.x) + 
			(gid.y + seSize.y) * imageSize.x] =  sub_sat(maxval, minval);
	}
}


#ifndef COORDS_SIZE
#define COORDS_SIZE 4
#endif

__kernel void gradient_c4_local_pragma(
	__global uchar* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cacheToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		uchar minval = erodeINF;
		uchar maxval = dilateINF;
		int c2 = COORDS_SIZE >> 1;
		
		#pragma unroll
		for(int i = 0; i < c2; ++i)
		{
			int4 coord = coords[i] + (int4)(lid, lid);

			uchar v0 = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			uchar v1 = sharedBlock[mad24(coord.w, sharedSize.x, coord.z)];
			
			minval = min(minval, v0);
			minval = min(minval, v1);
			maxval = max(maxval, v0);
			maxval = max(maxval, v1);
		}
		
		if(COORDS_SIZE % 2)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[COORDS_SIZE-1] + lid;
			
			uchar v = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			
			minval = min(minval, v);
			maxval = max(maxval, v);
		}
		
		output[(gid.x + seSize.x) + 
			(gid.y + seSize.y) * imageSize.x] = sub_sat(maxval, minval);	
	}
}

// ################################################################################
// Kernele dla el. strukturalnych o max. promieniu 14 (29x29)

__kernel
__attribute__((reqd_work_group_size(16,16,1))) 
void gradient4_local(
	__global uchar4* input,
	__global uchar* output,
	__constant int2* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cache4ToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		uchar minval = erodeINF;
		uchar maxval = dilateINF;

		for(int i = 0; i < seSize.z; ++i)
		{
			int2 coord = coords[i] + lid;
			uchar v = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			
			minval = min(minval, v);
			maxval = max(maxval, v);
		}
		
		output[(gid.x + seSize.x) +
			(gid.y + seSize.y)* imageSize.x] = sub_sat(maxval, minval);
	}
}

__kernel
__attribute__((reqd_work_group_size(16,16,1))) 
void gradient4_c4_local(
	__global uchar4* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cache4ToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		uchar minval = erodeINF;
		uchar maxval = dilateINF;
		int c2 = seSize.z >> 1;
		
		for(int i = 0; i < c2; ++i)
		{
			int4 coord = coords[i] + (int4)(lid, lid);
			
			uchar v0 = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			uchar v1 = sharedBlock[mad24(coord.w, sharedSize.x, coord.z)];
			
			minval = min(minval, v0);
			minval = min(minval, v1);
			maxval = max(maxval, v0);
			maxval = max(maxval, v1);
		}
		
		if(seSize.z % 2)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[seSize.z-1] + lid;
			
			uchar v = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			
			minval = min(minval, v);
			maxval = max(maxval, v);
		}
		
		output[(gid.x + seSize.x) +
			(gid.y + seSize.y) * imageSize.x] = sub_sat(maxval, minval);	
	}
}

__kernel
__attribute__((reqd_work_group_size(16,16,1))) 
void gradient4_c4_local_unroll(
	__global uchar4* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cache4ToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);

	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		uchar minval = erodeINF;
		uchar maxval = dilateINF;
		int c2 = seSize.z >> 1;
		int i = 0;
		
		for(; i < c2; i += 2)
		{
			int4 coord0 = coords[i]   + (int4)(lid, lid);
			int4 coord1 = coords[i+1] + (int4)(lid, lid);
			
			uchar v0 = sharedBlock[mad24(coord0.y, sharedSize.x, coord0.x)];
			uchar v1 = sharedBlock[mad24(coord0.w, sharedSize.x, coord0.z)];
			uchar v2 = sharedBlock[mad24(coord1.y, sharedSize.x, coord1.x)];
			uchar v3 = sharedBlock[mad24(coord1.w, sharedSize.x, coord1.z)];
			
			minval = min(minval, v0);
			minval = min(minval, v1);
			minval = min(minval, v2);
			minval = min(minval, v3);
			
			maxval = max(maxval, v0);
			maxval = max(maxval, v1);
			maxval = max(maxval, v2);
			maxval = max(maxval, v3);
		}
		
		i *= 2;
		
		for( ; i < seSize.z; ++i)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[i] + lid;
			
			uchar v = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			
			minval = min(minval, v);
			maxval = max(maxval, v);			
		}
		
		output[(gid.x + seSize.x) + 
			(gid.y + seSize.y) * imageSize.x] = sub_sat(maxval, minval);	
	}
}

__kernel
__attribute__((work_group_size_hint(16,16,1))) 
void gradient4_c4_local_pragma(
	__global uchar4* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = { get_global_id(0), get_global_id(1) };
	int2 lid = { get_local_id(0), get_local_id(1) };
	
	cache4ToLocalMemory(input, imageSize, lid, sharedSize, sharedBlock);
	
	if (gid.y < imageSize.y - mul24(seSize.y, 2) &&
		gid.x < imageSize.x - mul24(seSize.x, 2))
	{
		uchar minval = erodeINF;
		uchar maxval = dilateINF;
		int c2 = COORDS_SIZE / 2;
		
		#pragma unroll
		for(int i = 0; i < c2; ++i)
		{
			int4 coord = coords[i] + (int4)(lid, lid);
			
			uchar v0 = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			uchar v1 = sharedBlock[mad24(coord.w, sharedSize.x, coord.z)];
			
			minval = min(minval, v0);
			minval = min(minval, v1);
			maxval = max(maxval, v0);
			maxval = max(maxval, v1);
		}
		
		if(COORDS_SIZE % 2)
		{
			__constant int2* c = (__constant int2*)(coords);
			int2 coord = c[COORDS_SIZE-1] + lid;
			
			uchar v = sharedBlock[mad24(coord.y, sharedSize.x, coord.x)];
			
			minval = min(minval, v);
			maxval = max(maxval, v);
		}
		
		output[(gid.x + seSize.x) + 
			(gid.y + seSize.y) * imageSize.x] = sub_sat(maxval, minval);	
	}
}