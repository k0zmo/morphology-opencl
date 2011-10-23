__constant uint OBJ = 255;
__constant uint BCK = 0;

__kernel void outline(
	__read_only image2d_t src,
	__write_only image2d_t dst)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
	
	uint v1 = read_imageui(src, smp, gid + (int2)(-1, -1)).x;
	uint v2 = read_imageui(src, smp, gid + (int2)( 0, -1)).x;
	uint v3 = read_imageui(src, smp, gid + (int2)( 1, -1)).x;
	uint v4 = read_imageui(src, smp, gid + (int2)(-1,  0)).x;
	// v5
	uint v6 = read_imageui(src, smp, gid + (int2)( 1,  0)).x;
	uint v7 = read_imageui(src, smp, gid + (int2)(-1,  1)).x;
	uint v8 = read_imageui(src, smp, gid + (int2)( 0,  1)).x;
	uint v9 = read_imageui(src, smp, gid + (int2)( 1,  1)).x;
	
	if (v1 == OBJ &&
		v2 == OBJ &&
		v3 == OBJ &&
		v4 == OBJ &&
		v6 == OBJ &&
		v7 == OBJ &&
		v8 == OBJ &&
		v9 == OBJ)
	{
		write_imageui(dst, gid, (uint4)(BCK));
	}
}