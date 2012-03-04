#include "common.cl"

__constant uint4 shift = { 0, 2, 2, 0 };
__constant uint4 ddiv = { 2, 5, 2, 1 };
__constant uint4 coeff = { 4899, 9617, 1864, 0 };

uchar4 convert_bayer2rgb(
	__global uchar* src,
	const int2 size,
	int2 gid,
	bool x_odd, bool y_odd)
{
	uchar3 row1 = {
		src[(gid.x - 1) + (gid.y - 1) * size.x],
		src[(gid.x    ) + (gid.y - 1) * size.x],
		src[(gid.x + 1) + (gid.y - 1) * size.x],
	};

	uchar3 row2 = {
		src[(gid.x - 1) + (gid.y) * size.x],
		src[(gid.x    ) + (gid.y) * size.x],
		src[(gid.x + 1) + (gid.y) * size.x],
	};

	uchar3 row3 = {
		src[(gid.x - 1) + (gid.y + 1) * size.x],
		src[(gid.x    ) + (gid.y + 1) * size.x],
		src[(gid.x + 1) + (gid.y + 1) * size.x],
	};

	// Dla postaci gdzie G jest 4
	uint r = row2.y;
	uint g = row1.y + row2.x + row2.z + row3.y;
	uint b = row1.x + row1.z + row3.x + row3.z;

 	// Dla postaci gdzie G jest 5 a reszty po 2
	uint rr = row2.x + row2.z;
	uint gg = row1.x + row1.z + row2.y + row3.x + row3.z;
	uint bb = row1.y + row3.y;
	
	uint4 dd1 = (uint4)(r, g, b, 0) >> shift;
	uint4 dd2 = (uint4)(rr, gg, bb, 0) / ddiv;
	
	uchar4 out1 = convert_uchar4(dd1);
	uchar4 out2 = convert_uchar4(dd2);

 	uchar4 out = x_odd ?
 		(y_odd ? out1.xyzw : out2.zyxw) :
 		(y_odd ? out2.xyzw : out1.zyxw);
		
	return out;
}

#define DEFINE_BAYER_KERNEL_RGB(name, xo, yo) \
	__kernel void name( \
		__global uchar* src, \
		__global uchar4* dst, \
		const int2 size) \
	{ \
		int2 gid = { get_global_id(0), get_global_id(1) }; \
	\
		if(gid.x + 1 >= size.x || gid.y + 1 >= size.y) \
			return; \
	\
		bool x_odd = gid.x & 0x01; \
		bool y_odd = gid.y & 0x01; \
	\
		uchar4 out = convert_bayer2rgb(src, size, gid, xo(x_odd), yo(y_odd)); \
		dst[gid.x + gid.y * size.x] = out;	\
	}

#define DESCALE(x, n) (((x) + (1 << ((n)+1))) >> (n))
#define DEFINE_BAYER_KERNEL_GRAY(name, xo, yo) \
	__kernel void name( \
		__global uchar* src, \
		__global uchar* dst, \
		const int2 size) \
	{ \
		int2 gid = { get_global_id(0), get_global_id(1) }; \
	\
		if(gid.x + 1 >= size.x || gid.y + 1 >= size.y) \
			return; \
	\
		bool x_odd = gid.x & 0x01; \
		bool y_odd = gid.y & 0x01; \
	\
		uchar4 o = convert_bayer2rgb(src, size, gid, xo(x_odd), yo(y_odd)); \
		uint4 out = convert_uint4(o) * coeff; \
		uchar gray = convert_uchar_sat(DESCALE(out.x + out.y + out.z, 14)); \
		dst[gid.x + gid.y * size.x] = gray;	\
	}

DEFINE_BAYER_KERNEL_RGB(convert_rg2rgb, opTrue,  opTrue)
DEFINE_BAYER_KERNEL_RGB(convert_gb2rgb, opTrue,  opFalse)
DEFINE_BAYER_KERNEL_RGB(convert_gr2rgb, opFalse, opTrue)
DEFINE_BAYER_KERNEL_RGB(convert_bg2rgb, opFalse, opFalse)

DEFINE_BAYER_KERNEL_GRAY(convert_rg2gray, opTrue,  opTrue)
DEFINE_BAYER_KERNEL_GRAY(convert_gb2gray, opTrue,  opFalse)
DEFINE_BAYER_KERNEL_GRAY(convert_gr2gray, opFalse, opTrue)
DEFINE_BAYER_KERNEL_GRAY(convert_bg2gray, opFalse, opFalse)