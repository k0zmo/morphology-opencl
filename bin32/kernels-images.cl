__constant uchar erodeINF = 255;
__constant uchar dilateINF = 0;
__constant uchar OBJ = 255;
__constant uchar BCK = 0;

// dst = saturate((int)a - (int)b)

__kernel void subtract(
	__read_only image2d_t a,
	__read_only image2d_t b,
	__write_only image2d_t dst)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
	//int xsize = get_image_width(a);
	//int ysize = get_image_height(a);
	
	//if(gx >= xsize || gy >= ysize)
	//	return;
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
	int2 coords = (int2)(gx, gy);
		
	uchar pixa = read_imageui(a, smp, coords).x;
	uchar pixb = read_imageui(b, smp, coords).x;
	uchar pix = (pixb > pixa) ? (0) : (pixa - pixb);
	write_imageui(dst, coords, pix);
}

__kernel void addHalf(
	__read_only image2d_t skeleton,
	__read_only image2d_t src,
	__write_only image2d_t dst)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
	int2 coords = { gx, gy };
	
	uchar pix_skeleton = read_imageui(skeleton, smp, coords).x;
	uchar pix_src = read_imageui(src, smp, coords).x;
	
	if(pix_skeleton == BCK)
		write_imageui(dst, coords, pix_src / 2);
}

__kernel void erode(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant int2* coords,
	const int coords_size)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int2 gid = (int2)(gx, gy);
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
	uchar val = erodeINF;
	
	for(int i = 0; i < coords_size; ++i)
	{
		int2 coord = coords[i] + gid;
		val = min(val, read_imageui(src, smp, coord).x);
	}
	
	write_imageui(dst, gid, (uint4)(val));
}

__kernel void dilate(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant int2* coords,
	const int coords_size)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int2 gid = (int2)(gx, gy);
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
	uchar val = dilateINF;
	
	for(int i = 0; i < coords_size; ++i)
	{
		int2 coord = coords[i] + gid;
		val = max(val, read_imageui(src, smp, coord).x);
	}
	
	write_imageui(dst, gid, (uint4)(val));
}


__kernel void remove(
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