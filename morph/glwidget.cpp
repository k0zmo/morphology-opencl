#include "glwidget.h"
#include <QMessageBox>

GLWidget::GLWidget(QWidget* parent)
	: QGLWidget(parent), 
	prog(new QGLShaderProgram),
	swidth(-1), sheight(-1)
{

}
// -------------------------------------------------------------------------
GLWidget::~GLWidget()
{
	glDeleteTextures(1, &surface);
	glDeleteBuffers(1, &vboQuad);
}
// -------------------------------------------------------------------------
void GLWidget::initializeGL()
{
	glewInit();

	glDisable(GL_DEPTH_TEST);
	glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	// -------------------------------
	// Tekstura
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &surface);
	glBindTexture(GL_TEXTURE_2D, surface);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ONE);
	//glEnable(GL_TEXTURE_2D);

	// -------------------------------
	// Dane wierzcholkow
	struct Vertex { float x, y, s, t; };
	Vertex vertices[3] = { 
		{ -1, -1,   0, 2 },
		{  3, -1,   2, 2 },
		{ -1,  3,   0, 0 }
	};

	glGenBuffers(1, &vboQuad);
	glBindBuffer(GL_ARRAY_BUFFER, vboQuad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Pozycja
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
		sizeof(Vertex), (GLubyte*)nullptr);
	glEnableVertexAttribArray(0);

	// Koordynaty tekstury
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 
		sizeof(Vertex), (GLubyte*)nullptr + sizeof(float)*2);
	glEnableVertexAttribArray(1);

	//glVertexPointer(2, GL_FLOAT, sizeof(Vertex), (GLubyte*)nullptr);
	//glEnableClientState(GL_VERTEX_ARRAY);
	//glTexCoordPointer(2, GL_FLOAT, sizeof(Vertex), (GLubyte*)nullptr + sizeof(float)*2);
	//glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	// -------------------------------
	// Shadery
	auto raportError = [this]()
	{
		QMessageBox::critical(nullptr,
			"Critical error", prog->log(), QMessageBox::Ok);
		exit(-1);
	};

	if(!prog->addShaderFromSourceFile
		(QGLShader::Vertex, QLatin1String("gl/vertex.glsl")))
		raportError();

	if(!prog->addShaderFromSourceFile
		(QGLShader::Fragment, QLatin1String("gl/fragment.glsl")))
		raportError();

	if(!prog->link())
		raportError();

	prog->bind();
	prog->setUniformValue("surface", 0);
	prog->bindAttributeLocation("in_pos", 0);
	prog->bindAttributeLocation("in_texCoord", 1);

	//glUseProgram(0);
}
// -------------------------------------------------------------------------
void GLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawArrays(GL_TRIANGLES, 0, 3);

	GLenum err = glGetError();
	if(err) printf("OpenGL Error: 0x0%x\n", err);
}
// -------------------------------------------------------------------------
void GLWidget::resizeGL(int width, int height)
{
	if(height == 0)
		height = 1;
	glViewport(0, 0, width, height);
}
// -------------------------------------------------------------------------
void GLWidget::setSurface(const cv::Mat& cvSurface)
{
	makeCurrent();
	createSurface_impl(cvSurface.cols, cvSurface.rows, cvSurface.data);
	updateGL();
}
// -------------------------------------------------------------------------
GLuint GLWidget::createEmptySurface(int w, int h)
{
	createSurface_impl(w, h, nullptr);
	return surface;
}
// -------------------------------------------------------------------------
void GLWidget::createSurface_impl(int w, int h, const void* data)
{
	makeCurrent();

	if(swidth != w || sheight != h)
	{
		swidth = w;
		sheight = h;

		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, swidth, 
			sheight, 0, GL_RED, GL_UNSIGNED_BYTE, data);
	}
	else
	{
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, swidth,
			sheight, GL_RED, GL_UNSIGNED_BYTE, data);
	}
}
