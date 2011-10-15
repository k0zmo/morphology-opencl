#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uchar erodeINF = 255;

__kernel void erode(
	__global uchar* input,
	__global uchar* output,
	__constant int2* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	uchar val = erodeINF;
	
	for(int i = 0; i < seSize.z; ++i)
	{
		int2 coord = coords[i] + gid;
		val = min(val, input[coord.x + coord.y * imageSize.x]);
	}
	
	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;
}

__kernel void erode_c4(
	__global uchar* input,
	__global uchar* output,
	__constant int4* coords,
	const int4 seSize, // { kradiusX, kradiusY, coords.size() }
	const int2 imageSize)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	uchar val = erodeINF;
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
	uchar val = erodeINF;
	for(int i = 0; i < seSize.z; ++i)
	{
		int2 coord = coords[i] + lid;
		val = min(val, sharedBlock[coord.x + coord.y * sharedSize.x]);
	}
	
	output[(gid.x + seSize.x) + (gid.y + seSize.y)* imageSize.x] = val;
}