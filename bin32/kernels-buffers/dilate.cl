#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uchar dilateINF = 0;

__kernel void dilate(
	__global uchar* input,
	__global uchar* output,
	__constant int2* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	uchar val = dilateINF;
	
	for(int i = 0; i < seSize.z; ++i)
	{
		int2 coord = coords[i] + gid;
		val = max(val, input[coord.x + coord.y * imageSize.x]);
	}
	
	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;
}

__kernel void dilate_c4(
	__global uchar* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	uchar val = dilateINF;
	int c2 = seSize.z >> 1;
	
	for(int i = 0; i < c2; ++i)
	{
		int4 coord = coords[i] + (int4)(gid, gid);
		val = max(val, input[coord.x + coord.y * imageSize.x]);
		val = max(val, input[coord.z + coord.w * imageSize.x]);
	}
	
	if(seSize.z % 2)
	{
		__constant int2* c = (__constant int2*)(coords);
		int2 coord = c[seSize.z-1] + gid;
		val = max(val, input[coord.x + coord.y * imageSize.x]);
	}

	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;
}

__kernel void dilate_local(
	__global uchar* input,
	__global uchar* output,
	__constant int2* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	int2 localSize = (int2)(get_local_size(0), get_local_size(1));
	int2 groupId = (int2)(get_group_id(0), get_group_id(1));
	int2 groupStartId = groupId * localSize; // id pierwszego bajtu w tej grupie roboczej
		
	// Zaladuj obszar roboczy obrazu zrodlowego do pamieci lokalnej
	for(int y = lid.y; y < sharedSize.y; y += localSize.y)
	{
		int r = groupStartId.y + y; // indeks.y bajtu z wejsca
		for(int x = lid.x; x < sharedSize.x; x += localSize.x)
		{
			int c = groupStartId.x + x; // indeks.x bajtu z wejscia
			
			if(c < imageSize.x && r < imageSize.y)
			{
				sharedBlock[x + y * sharedSize.x] = input[c + r * imageSize.x];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - seSize.y*2)
		return;
		
	if(gid.x >= imageSize.x - seSize.x*2)
		return;
	
	// Filtracja wlasciwa
	uchar val = dilateINF;
	for(int i = 0; i < seSize.z; ++i)
	{
		int2 coord = coords[i] + lid;
		val = max(val, sharedBlock[coord.x + coord.y * sharedSize.x]);
	}
	
	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;
}

__kernel void dilate_c4_local(
	__global uchar* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	int2 lid = (int2)(get_local_id(0), get_local_id(1));
	int2 localSize = (int2)(get_local_size(0), get_local_size(1));
	int2 groupId = (int2)(get_group_id(0), get_group_id(1));
	int2 groupStartId = groupId * localSize; // id pierwszego bajtu w tej grupie roboczej
		
	// Zaladuj obszar roboczy obrazu zrodlowego do pamieci lokalnej
	for(int y = lid.y; y < sharedSize.y; y += localSize.y)
	{
		int r = groupStartId.y + y; // indeks.y bajtu z wejsca
		for(int x = lid.x; x < sharedSize.x; x += localSize.x)
		{
			int c = groupStartId.x + x; // indeks.x bajtu z wejscia
			
			if(c < imageSize.x && r < imageSize.y)
			{
				sharedBlock[x + y * sharedSize.x] = input[c + r * imageSize.x];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - seSize.y*2)
		return;
		
	if(gid.x >= imageSize.x - seSize.x*2)
		return;
	
	// Filtracja wlasciwa
	uchar val = dilateINF;
	int c2 = seSize.z >> 1;
	
	for(int i = 0; i < c2; ++i)
	{
		int4 coord = coords[i] + (int4)(lid, lid);
		val = max(val, sharedBlock[coord.x + coord.y * sharedSize.x]);
		val = max(val, sharedBlock[coord.z + coord.w * sharedSize.x]);
	}
	
	if(seSize.z % 2)
	{
		__constant int2* c = (__constant int2*)(coords);
		int2 coord = c[seSize.z-1] + lid;
		val = max(val, sharedBlock[coord.x + coord.y * imageSize.x]);
	}
	
	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;
}

__kernel __attribute__((reqd_work_group_size(16,16,1))) 
void dilate4_local(
	__global uchar4* input,
	__global uchar* output,
	__constant int2* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 localSize = (int2)(get_local_size(0), get_local_size(1));
	int2 groupId = (int2)(get_group_id(0), get_group_id(1));
	int2 groupStartId = groupId * localSize; // id pierwszego bajtu w tej grupie roboczej
	
	// Przebiega od 0 do 255
	int flatLid = get_local_id(0) + get_local_id(1) * localSize.x;
	
	int2 lid;
	lid.x = (flatLid % (sharedSize.x/4));
	lid.y = (flatLid / (sharedSize.x/4));	
	
	int2 gid;
	gid.x = groupStartId.x/4 + lid.x;
	gid.y = groupStartId.y   + lid.y;
		
	__local uchar4* sharedBlock4 = (__local uchar4*)(&sharedBlock[lid.x*4 + lid.y*sharedSize.x]);
	
	if (gid.y < imageSize.y && 
		gid.x < imageSize.x/4 && 
		lid.y < sharedSize.y)
	{
		sharedBlock4[0] = input[gid.x + gid.y*imageSize.x/4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	gid = (int2)(get_global_id(0), get_global_id(1));
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - seSize.y*2)
		return;
		
	if(gid.x >= imageSize.x - seSize.x*2)
		return;
		
	lid = (int2)(get_local_id(0), get_local_id(1));
	
	// Filtracja wlasciwa
	uchar val = dilateINF;

	for(int i = 0; i < seSize.z; ++i)
	{
		int2 coord = coords[i] + lid;
		val = max(val, sharedBlock[coord.x + coord.y * sharedSize.x]);
	}
	
	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;
}

__kernel __attribute__((reqd_work_group_size(16,16,1))) 
void dilate4_c4_local(
	__global uchar4* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 localSize = (int2)(get_local_size(0), get_local_size(1));
	int2 groupId = (int2)(get_group_id(0), get_group_id(1));
	int2 groupStartId = groupId * localSize; // id pierwszego bajtu w tej grupie roboczej
	
	// Przebiega od 0 do 255
	int flatLid = get_local_id(0) + get_local_id(1) * localSize.x;
	
	int2 lid;
	lid.x = (flatLid % (sharedSize.x/4));
	lid.y = (flatLid / (sharedSize.x/4));	
	
	int2 gid;
	gid.x = groupStartId.x/4 + lid.x;
	gid.y = groupStartId.y   + lid.y;
		
	__local uchar4* sharedBlock4 = (__local uchar4*)(&sharedBlock[lid.x*4 + lid.y*sharedSize.x]);
	
	if (gid.y < imageSize.y && 
		gid.x < imageSize.x/4 && 
		lid.y < sharedSize.y)
	{
		sharedBlock4[0] = input[gid.x + gid.y*imageSize.x/4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	gid = (int2)(get_global_id(0), get_global_id(1));
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - seSize.y*2)
		return;
		
	if(gid.x >= imageSize.x - seSize.x*2)
		return;
		
	lid = (int2)(get_local_id(0), get_local_id(1));
	
	// Filtracja wlasciwa
	uchar val = dilateINF;	
	int c2 = seSize.z >> 1;
	
	for(int i = 0; i < c2; ++i)
	{
		int4 coord = coords[i] + (int4)(lid, lid);
		val = max(val, sharedBlock[coord.x + coord.y * sharedSize.x]);
		val = max(val, sharedBlock[coord.z + coord.w * sharedSize.x]);
	}
	
	if(seSize.z % 2)
	{
		__constant int2* c = (__constant int2*)(coords);
		int2 coord = c[seSize.z-1] + lid;
		val = max(val, sharedBlock[coord.x + coord.y * imageSize.x]);
	}
	
	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;	
}


#ifndef COORDS_SIZE
#define COORDS_SIZE 169
#endif

__kernel __attribute__((work_group_size_hint(16,16,1))) 
void dilate4_c4_local_def(
	__global uchar4* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize,
	__local uchar* sharedBlock,
	const int2 sharedSize) // { sharedBlockSizeX, sharedBlockSizeY }
{
	int2 localSize = (int2)(get_local_size(0), get_local_size(1));
	int2 groupId = (int2)(get_group_id(0), get_group_id(1));
	int2 groupStartId = groupId * localSize; // id pierwszego bajtu w tej grupie roboczej
	
	// Przebiega od 0 do 255
	int flatLid = get_local_id(0) + get_local_id(1) * localSize.x;
	
	int2 lid;
	lid.x = (flatLid % (sharedSize.x/4));
	lid.y = (flatLid / (sharedSize.x/4));	
	
	int2 gid;
	gid.x = groupStartId.x/4 + lid.x;
	gid.y = groupStartId.y   + lid.y;
		
	__local uchar4* sharedBlock4 = (__local uchar4*)(&sharedBlock[lid.x*4 + lid.y*sharedSize.x]);
	
	if (gid.y < imageSize.y && 
		gid.x < imageSize.x/4 && 
		lid.y < sharedSize.y)
	{
		sharedBlock4[0] = input[gid.x + gid.y*imageSize.x/4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	gid = (int2)(get_global_id(0), get_global_id(1));
	
	// Poniewaz NDRange jest wielokrotnoscia rozmiaru localSize
	// musimy sprawdzic ponizsze warunki
	if(gid.y >= imageSize.y - seSize.y*2)
		return;
		
	if(gid.x >= imageSize.x - seSize.x*2)
		return;
		
	lid = (int2)(get_local_id(0), get_local_id(1));
	
	// Filtracja wlasciwa
	uchar val = dilateINF;	
	int c2 = COORDS_SIZE / 2;
	
	#pragma unroll
	for(int i = 0; i < c2; ++i)
	{
		int4 coord = coords[i] + (int4)(lid, lid);
		val = max(val, sharedBlock[coord.x + coord.y * sharedSize.x]);
		val = max(val, sharedBlock[coord.z + coord.w * sharedSize.x]);
	}
	
	if(COORDS_SIZE % 2)
	{
		__constant int2* c = (__constant int2*)(coords);
		int2 coord = c[seSize.z-1] + lid;
		val = max(val, sharedBlock[coord.x + coord.y * imageSize.x]);
	}
	
	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;	
}