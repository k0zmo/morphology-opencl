#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uint dilateINF = 0;
__constant uint erodeINF = 255;
__constant uint OBJ = 255;
__constant uint BCK = 0;

#ifdef USE_ATOMIC_COUNTERS
#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable 
#define counter_type counter32_t
#else
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define counter_type __global uint*
#define atomic_inc atom_inc
#endif

//
// Cache'uje dane z pamieci globalnej do lokalnej 
//

__attribute__((always_inline))
void cacheToLocalMemory(
	__global uint* input,
	const int2 imageSize,
	const int2 lid,
	const int2 sharedSize,
	__local uint* sharedBlock)
{
	int2 localSize = { get_local_size(0), get_local_size(1) };
	int2 groupId = { get_group_id(0), get_group_id(1) };
	int2 groupStartId = mul24(groupId, localSize); // id pierwszego bajtu w tej grupie roboczej
		
	// Zaladuj obszar roboczy obrazu zrodlowego do pamieci lokalnej
	for(int y = lid.y; y < sharedSize.y; y += localSize.y)
	{
		int r = groupStartId.y + y; // indeks.y bajtu z wejsca
		for(int x = lid.x; x < sharedSize.x; x += localSize.x)
		{
			int c = groupStartId.x + x; // indeks.x bajtu z wejscia
			
			if(c < imageSize.x && r < imageSize.y)
			{
				sharedBlock[mad24(y, sharedSize.x, x)] = input[c + r * imageSize.x];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

//
// Cache'uje dane z pamieci globalnej do lokalnej w paczkach po 4
// Dziala tylko dla kerneli o rozmiarze max 17x17
//

__attribute__((always_inline))
void cache4ToLocalMemory(
	__global uint4* input,
	const int2 imageSize,
	const int2 lid,
	const int2 sharedSize,
	__local uint* sharedBlock)
{
	int2 localSize = { get_local_size(0), get_local_size(1) };
	int2 groupId = { get_group_id(0), get_group_id(1) };
	int2 groupStartId = mul24(groupId, localSize); // id pierwszego bajtu w tej grupie roboczej
	
	int sharedSizex4 = sharedSize.x/4;
	int imageSizex4 = imageSize.x/4;
	
	// Przebiega od 0 do 255
	int flatLid = mad24(lid.y, localSize.x, lid.x);
	
	int2 tid = (int2) {
		flatLid % (sharedSizex4),
		flatLid / (sharedSizex4)
	};
	
	int2 gid = (int2) {
		groupStartId.x/4 + tid.x,
		groupStartId.y   + tid.y
	};
	
	if (gid.y < imageSize.y && 
		//gid.x < imageSize.x/4 && 
		tid.y < sharedSize.y)
	{
		__local uint4* sharedBlock4 = (__local uint4*)(&sharedBlock[mad24(tid.y, sharedSize.x, tid.x*4)]);
		sharedBlock4[0] = input[gid.x + gid.y*imageSize.x/4];
		
		// jesli chcemy wczytac dodatkowy wektor
		
		int lsize = mul24(localSize.x, localSize.y);
		int ssize = mul24(sharedSize.x, sharedSize.y);
		
		if((flatLid+lsize)*4 < ssize)
		{
			int2 gid = (int2) {
				groupStartId.x/4 + (flatLid+lsize) % (sharedSizex4),
				groupStartId.y   + (flatLid+lsize) / (sharedSizex4)
			};
			
			sharedBlock4[lsize] = input[gid.x + gid.y*imageSizex4];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

//
// To samo co funkcja wyzej, z tym, �e sharedSize jest stala zdefiniowana w czasie kompilacji
// Uzywana tam gdzie el. strukturalny jest stalego rozmiaru 3x3 (szkieletyzacja, hit-miss)
//

#define SHARED_SIZEX 20
#define SHARED_SIZEY 18

__attribute__((always_inline))
void cache4ToLocalMemory16(
	__global uint4* input,
	const int2 imageSize,
	const int2 lid,
	__local uint sharedBlock[SHARED_SIZEY][SHARED_SIZEX])
{
	int2 localSize = { get_local_size(0), get_local_size(1) };
	int2 groupId = { get_group_id(0), get_group_id(1) };
	int2 groupStartId = mul24(groupId, localSize); // id pierwszego bajtu w tej grupie roboczej
	
	// Przebiega od 0 do 255
	int flatLid = mad24(lid.y, localSize.x, lid.x);
	
	int2 tid = (int2)(
		flatLid % (SHARED_SIZEX/4),
		flatLid / (SHARED_SIZEX/4));	
	
	int2 gid = (int2)(
		groupStartId.x/4 + tid.x,
		groupStartId.y   + tid.y);	
	
	if (gid.y < imageSize.y && 
		gid.x < imageSize.x/4 && 
		tid.y < SHARED_SIZEY)
	{
		__local uint4* sharedBlock4 = (__local uint4*)(&sharedBlock[tid.y][tid.x*4]);
		sharedBlock4[0] = input[gid.x + gid.y*imageSize.x/4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}