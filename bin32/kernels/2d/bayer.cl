#include "common.cl"

__attribute__((always_inline)) bool opTrue(bool o) { return o; }
__attribute__((always_inline)) bool opFalse(bool o) { return !o; }

__constant float4 greyscale = { 0.2989f, 0.5870f, 0.1140f, 0 };
__constant float4 mul1 = { 1, 0.25f, 0.25f, 1.0f };
__constant float4 mul2 = { 0.5f, 0.2f, 0.5f, 1.f };

// Funkcja pomocnicza dla 4 kerneli convert_{X}2rgb
// gdzie X = {rg, gr, gb, bg}
float4 convert_bayer2rgb(
	__read_only image2d_t src,
	int2 gid,
	bool x_odd, bool y_odd)
{
	// "Narozniki"
	float4 v1 = {
		read_imagef(src, smp, gid + (int2)(-1, -1)).x,
		read_imagef(src, smp, gid + (int2)( 1, -1)).x,
		read_imagef(src, smp, gid + (int2)(-1,  1)).x,
		read_imagef(src, smp, gid + (int2)( 1,  1)).x,
	};

	// "Romb"
	float4 v2 = {
		read_imagef(src, smp, gid + (int2)( 0, -1)).x,
		read_imagef(src, smp, gid + (int2)(-1,  0)).x,
		read_imagef(src, smp, gid + (int2)( 0,  1)).x,
		read_imagef(src, smp, gid + (int2)( 1,  0)).x,
	};

	// "Srodek"
	float v3 = read_imagef(src, smp, gid).x;

	// Dla postaci gdzie G jest 4
	float2 b2 = v1.xy + v1.zw;
	float b = b2.x + b2.y;
	float2 g2 = v2.xy + v2.zw;
	float g = g2.x + g2.y;
	float r = v3;

	float4 out1 = {r, g, b, 1};
	out1 *= mul1;

	// Dla postaci gdzie G jest 5 a reszty po 2
	float rr = v2.y + v2.w;
	float bb = v2.x + v2.z;
	float gg = b2.x + b2.y + v3;

	float4 out2 = (float4)(rr, gg, bb, 1);
	out2 *= mul2;
	
	float4 out = x_odd ?
		(y_odd ? out1.xyzw : out2.zyxw) :
		(y_odd ? out2.xyzw : out1.zyxw);	

	return out;
}

#define DEFINE_BAYER_KERNEL_RGB(name, xo, yo) \
	__kernel void name( \
		__read_only image2d_t src, \
		__write_only image2d_t dst) \
	{ \
		int2 gid = (int2) { get_global_id(0), get_global_id(1) }; \
		int2 size = (int2) { get_image_width(src), get_image_height(src) }; \
	\
		if(gid.x + 1 >= size.x || gid.y + 1 >= size.y) \
			return; \
	\
		bool x_odd = gid.x & 0x01; \
		bool y_odd = gid.y & 0x01; \
	\
		float4 out = convert_bayer2rgb(src, gid, xo(x_odd), yo(y_odd)); \
		write_imagef(dst, gid, out); \
	}
	
#define DEFINE_BAYER_KERNEL_GRAY(name, xo, yo) \
	__kernel void name( \
		__read_only image2d_t src, \
		__write_only image2d_t dst) \
	{ \
		int2 gid = (int2) { get_global_id(0), get_global_id(1) }; \
		int2 size = (int2) { get_image_width(src), get_image_height(src) }; \
	\
		if(gid.x + 1 >= size.x || gid.y + 1 >= size.y) \
			return; \
	\
		bool x_odd = gid.x & 0x01; \
		bool y_odd = gid.y & 0x01; \
	\
		float4 out = convert_bayer2rgb(src, gid, xo(x_odd), yo(y_odd)); \
		float vg = dot(greyscale, out); \
		out = (float4)(vg); \
		write_imagef(dst, gid, out); \
	}	
	
DEFINE_BAYER_KERNEL_RGB(convert_rg2rgb, opTrue,  opTrue)
DEFINE_BAYER_KERNEL_RGB(convert_gb2rgb, opTrue,  opFalse)
DEFINE_BAYER_KERNEL_RGB(convert_gr2rgb, opFalse, opTrue)
DEFINE_BAYER_KERNEL_RGB(convert_bg2rgb, opFalse, opFalse)

DEFINE_BAYER_KERNEL_GRAY(convert_rg2gray, opTrue,  opTrue)
DEFINE_BAYER_KERNEL_GRAY(convert_gb2gray, opTrue,  opFalse)
DEFINE_BAYER_KERNEL_GRAY(convert_gr2gray, opFalse, opTrue)
DEFINE_BAYER_KERNEL_GRAY(convert_bg2gray, opFalse, opFalse)
