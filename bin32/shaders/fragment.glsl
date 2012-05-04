varying vec2 v2f_texCoord;
uniform sampler2D surface;

void main()
{
	gl_FragColor = vec4(texture2D(surface, v2f_texCoord).rrr, 1);
}