#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__constant uchar erodeINF = 255;

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