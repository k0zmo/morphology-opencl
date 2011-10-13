__constant uchar erodeINF = 255;

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
		
	uint val = erodeINF;
	
	for(int i = 0; i < coords_size; ++i)
	{
		int2 coord = coords[i] + gid;	
		val = min(val, read_imageui(src, smp, coord).x);
	}
	
	write_imageui(dst, gid, (uint4)(val));
}

__kernel void erode_c4(
	__read_only image2d_t src,
	__write_only image2d_t dst,
	__constant int4* coords,
	const int coords_size)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int2 gid = (int2)(gx, gy);
	
	const sampler_t smp = 
		CLK_NORMALIZED_COORDS_FALSE | 
		CLK_FILTER_NEAREST | 
		CLK_ADDRESS_CLAMP_TO_EDGE;
		
	uint val = erodeINF;
	int c2 = coords_size >> 1;
	
	for(int i = 0; i < c2; ++i)
	{
		int4 g = (int4)(gid, gid);
		int4 coord = coords[i] + g;	
		
		val = min(val, read_imageui(src, smp, coord.xy).x);
		val = min(val, read_imageui(src, smp, coord.zw).x);
	}
	
	// Dla masek 2n+1 x 2m+1 ilosc wspolrzednych zawsze bedzie nieparzysta
	//if(coords_size % 2)
	{
		__constant int2* c = (__constant int2*)(coords);
		int2 coord = c[coords_size] + gid;
		val = min(val, read_imageui(src, smp, coord).x);
	}
	
	write_imageui(dst, gid, (uint4)(val));
}