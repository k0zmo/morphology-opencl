#define SHARED_SIZEX 20
#define SHARED_SIZEY 18

__attribute__((always_inline))
void cacheNeighbours(
	__global uint4* input,
	const int2 imageSize,
	__local uint sharedBlock[SHARED_SIZEY][SHARED_SIZEX])
{
	int2 localSize = (int2)(get_local_size(0), get_local_size(1));
	int2 groupId = (int2)(get_group_id(0), get_group_id(1));
	int2 groupStartId = groupId * localSize; // id pierwszego bajtu w tej grupie roboczej
	
	// Przebiega od 0 do 255
	int flatLid = get_local_id(0) + get_local_id(1) * localSize.x;
	
	int2 lid;
	lid.x = (flatLid % (SHARED_SIZEX/4));
	lid.y = (flatLid / (SHARED_SIZEX/4));	
	
	int2 gid;
	gid.x = groupStartId.x/4 + lid.x;
	gid.y = groupStartId.y   + lid.y;	
	
	__local uint4* sharedBlock4 = (__local uint4*)(&sharedBlock[lid.y][lid.x*4]);
	
	if (gid.y < imageSize.y && 
		gid.x < imageSize.x/4 && 
		lid.y < SHARED_SIZEY)
	{
		sharedBlock4[0] = input[gid.x + gid.y*imageSize.x/4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}