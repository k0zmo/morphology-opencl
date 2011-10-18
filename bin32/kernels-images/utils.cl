__constant uint OBJ = 255;
__constant uint BCK = 0;

// dst = saturate((int)a - (int)b)

__kernel void subtract(
	__read_only image2d_t a,
	__read_only image2d_t b,
	__write_only image2d_t dst)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
	int2 coords = (int2)(gx, gy);
		
	uint pixa = read_imageui(a, smp, coords).x;
	uint pixb = read_imageui(b, smp, coords).x;
	uint pix = sub_sat(pixa, pixb);
	
	write_imageui(dst, coords, pix);
}