// Dla adresowania uint8*
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void invert(__global unsigned char* input,
	__global unsigned char* output)
{
	size_t tid = get_global_id(0);
	
	output[tid] = 255 - input[tid];
}

__constant sampler_t sampler = 
	CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE | 
	CLK_FILTER_NEAREST;

__kernel void invertImage(__read_only image2d_t srcImage,
	__write_only image2d_t dstImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	uint4 color = read_imageui(srcImage, sampler, coord);
	color = (uint4)(255) - color;
	write_imageui(dstImage, coord, color);
}

__kernel void erode3x3(__global unsigned char* input,
	__global unsigned char* output)
{
	int2 pos = (int2)(get_global_id(0), get_global_id(1));
	
	size_t width = get_global_size(0);
	size_t height = get_global_size(1);
	size_t id = pos.x + pos.y * width; 
	
	if(pos.x == 0 || pos.y == 0 || pos.x >= width - 1 || pos.y >= height - 1)
	{
		output[id] = 0;
	}
	else
	{
		uint px[9];

		px[0] = input[id - 1 - width];
		px[1] = input[id - width];
		px[2] = input[id + 1 - width];
		px[3] = input[id - 1];
		px[4] = input[id];
		px[5] = input[id + 1];
		px[6] = input[id - 1 + width];
		px[7] = input[id + width];
		px[8] = input[id + 1 + width];
		
		uint val = min(min(min(min(min(min(min(min(px[0], px[1]), px[2]), px[3]), px[4]), px[5]), px[6]), px[7]), px[8]);
		
		output[id] = val;
	}
}

__kernel void erodeImage3x3(__read_only image2d_t srcImage,
	__write_only image2d_t dstImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	
	uint color1 = read_imageui(srcImage, sampler, (int2)(coord.x - 1, coord.y - 1)).s0;
	uint color2 = read_imageui(srcImage, sampler, (int2)(coord.x    , coord.y - 1)).s0;
	uint color3 = read_imageui(srcImage, sampler, (int2)(coord.x + 1, coord.y - 1)).s0;

	uint color4 = read_imageui(srcImage, sampler, (int2)(coord.x - 1, coord.y)).s0;
	uint color5 = read_imageui(srcImage, sampler, (int2)(coord.x    , coord.y)).s0;
	uint color6 = read_imageui(srcImage, sampler, (int2)(coord.x + 1, coord.y)).s0;

	uint color7 = read_imageui(srcImage, sampler, (int2)(coord.x - 1, coord.y + 1)).s0;
	uint color8 = read_imageui(srcImage, sampler, (int2)(coord.x    , coord.y + 1)).s0;
	uint color9 = read_imageui(srcImage, sampler, (int2)(coord.x + 1, coord.y + 1)).s0;
	
	uint mincolor = min(min(min(min(min(min(min(min(color1, color2), color3), color4), color5), color6), color7), color8), color9);
	uint4 mincolor4 = (uint4)(mincolor);
	
	write_imageui(dstImage, coord, mincolor4);
	
}