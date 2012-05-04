attribute vec2 in_pos;
attribute vec2 in_texCoord;
varying vec2 v2f_texCoord;

void main()
{
	gl_Position = vec4(in_pos, 0, 1);
	v2f_texCoord = in_texCoord;
}