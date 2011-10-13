__constant uchar OBJ = 255;
__constant uchar BCK = 0;

__kernel void thinning(
	__read_only image2d_t src,
	__write_only image2d_t dst)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int2 gid = (int2)(gx, gy);
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
	
	uchar v1 = read_imageui(src, smp, gid + (int2)(-1, -1)).x;
	uchar v2 = read_imageui(src, smp, gid + (int2)( 0, -1)).x;
	uchar v3 = read_imageui(src, smp, gid + (int2)( 1, -1)).x;
	uchar v4 = read_imageui(src, smp, gid + (int2)(-1,  0)).x;
	// v5
	uchar v6 = read_imageui(src, smp, gid + (int2)( 1,  0)).x;
	uchar v7 = read_imageui(src, smp, gid + (int2)(-1,  1)).x;
	uchar v8 = read_imageui(src, smp, gid + (int2)( 0,  1)).x;
	uchar v9 = read_imageui(src, smp, gid + (int2)( 1,  1)).x;
	
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