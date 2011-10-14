__constant uint OBJ = 255;
__constant uint BCK = 0;

__kernel void thinning(
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

__kernel void thinning_lut(
	__read_only image2d_t src,
	__write_only image2d_t dst)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
	
	uint v1 = read_imageui(src, smp, gid + (int2)(-1, -1)).x & 0x01;
	uint v2 = read_imageui(src, smp, gid + (int2)( 0, -1)).x & 0x01;
	uint v3 = read_imageui(src, smp, gid + (int2)( 1, -1)).x & 0x01;
	uint v4 = read_imageui(src, smp, gid + (int2)(-1,  0)).x & 0x01;
	// v5
	uint v6 = read_imageui(src, smp, gid + (int2)( 1,  0)).x & 0x01;
	uint v7 = read_imageui(src, smp, gid + (int2)(-1,  1)).x & 0x01;
	uint v8 = read_imageui(src, smp, gid + (int2)( 0,  1)).x & 0x01;
	uint v9 = read_imageui(src, smp, gid + (int2)( 1,  1)).x & 0x01;
	
	const uint w = (1 << 8) | (1 << 7) | (1 << 6) | 
		(1 << 5) | (1 << 3) | (1 << 2) | (1 << 1) | 1;

	uint v =
		(v1) | 
		(v2 << 1) |
		(v3 << 2) |
		(v4 << 3) |
		(v6 << 5) |
		(v7 << 6) |
		(v8 << 7) |
		(v9 << 8);
	
	if (v == w)
		write_imageui(dst, gid, (uint4)(BCK));
}

__kernel void thinning4(
	__read_only image2d_t src,
	__write_only image2d_t dst)
{
	int2 gid = (int2)(get_global_id(0), get_global_id(1));
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
	
	uint4 v1 = read_imageui(src, smp, gid + (int2)(-1, -1));
	uint4 v2 = read_imageui(src, smp, gid + (int2)( 0, -1));
	uint4 v3 = read_imageui(src, smp, gid + (int2)( 1, -1));
	uint4 v4 = read_imageui(src, smp, gid + (int2)(-1,  0));
	uint4 v5 = read_imageui(src, smp, gid + (int2)( 0 , 0));
	uint4 v6 = read_imageui(src, smp, gid + (int2)( 1,  0));
	uint4 v7 = read_imageui(src, smp, gid + (int2)(-1,  1));
	uint4 v8 = read_imageui(src, smp, gid + (int2)( 0,  1));
	uint4 v9 = read_imageui(src, smp, gid + (int2)( 1,  1));

	uint4 to_write = (uint4)(OBJ);
	
	if (v1.s3 == OBJ &&
		v2.s0 == OBJ &&
		v2.s1 == OBJ &&
		v4.s3 == OBJ &&
		v5.s1 == OBJ &&
		v7.s3 == OBJ &&
		v8.s0 == OBJ &&
		v8.s1 == OBJ)
	{
		to_write.s0 = BCK;
	}
	else
	{
		to_write.s0 = v5.s0;
	}

	if (v2.s0 == OBJ &&
		v2.s1 == OBJ &&
		v2.s2 == OBJ &&
		v5.s0 == OBJ &&
		v5.s2 == OBJ &&
		v8.s0 == OBJ &&
		v8.s1 == OBJ &&
		v8.s2 == OBJ)
	{
		to_write.s1 = BCK;
	}
	else
	{
		to_write.s1 = v5.s1;
	}

	if (v2.s1 == OBJ &&
		v2.s2 == OBJ &&
		v2.s3 == OBJ &&
		v5.s1 == OBJ &&
		v5.s3 == OBJ &&
		v8.s1 == OBJ &&
		v8.s2 == OBJ &&
		v8.s3 == OBJ)
	{
		to_write.s2 = BCK;
	}
	else
	{
		to_write.s2 = v5.s2;
	}

	if (v2.s2 == OBJ &&
		v2.s3 == OBJ &&
		v3.s0 == OBJ &&
		v5.s2 == OBJ &&
		v6.s0 == OBJ &&
		v8.s2 == OBJ &&
		v8.s3 == OBJ &&
		v9.s0 == OBJ)
	{
		to_write.s3 = BCK;
	}
	else
	{
		to_write.s3 = v5.s3;
	}

	write_imageui(dst, gid, to_write);
}