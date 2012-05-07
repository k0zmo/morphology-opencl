#include "glwidget.h"

GLDummyWidget::GLDummyWidget(QWidget* parent,
	const QGLWidget* shareWidget)
	: QGLWidget(parent, shareWidget)
	, d_surface(0)
	, d_width(-1)
	, d_height(-1)
	, deleteTexUponDestruction(false)
{
}

GLDummyWidget::~GLDummyWidget()
{
	if(d_surface && deleteTexUponDestruction)
		glDeleteTextures(1, &d_surface);
}

void GLDummyWidget::initializeWithNewSurface(int initWidth, int initHeight)
{
	if(d_surface)
		glDeleteTextures(1, &d_surface);

	glGenTextures(1, &d_surface);
	glBindTexture(GL_TEXTURE_2D, d_surface);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

	d_width = initWidth;
	d_height = initHeight;
	deleteTexUponDestruction = true;

	glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, d_width, d_height,
		0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

	initialize_impl();

	GLenum err = glGetError();
	if(err) printf("OpenGL Error: 0x0%x\n", err);
}

void GLDummyWidget::initializeWithSharedSurface(GLuint surface)
{
	if(d_surface)
		glDeleteTextures(1, &d_surface);

	d_surface = surface;
	d_width = -1;
	d_height = -1;
	deleteTexUponDestruction = false;

	initialize_impl();
}

void GLDummyWidget::setSurfaceData(const cv::Mat& cvSurface)
{
	createSurface_impl(cvSurface.cols, cvSurface.rows, cvSurface.data);
}

GLuint GLDummyWidget::resizeSurface(int w, int h)
{
	createSurface_impl(w, h, nullptr);
	return d_surface;
}

void GLDummyWidget::createSurface_impl(int w, int h, const void* data)
{
	if(!d_surface)
		return;

	//qDebug() << "         <" << __FUNCTION__ ">";
	//makeCurrent();

	glBindTexture(GL_TEXTURE_2D, d_surface);

	// Nastapila zmiana rozmiaru - alokujemy pamiec od nowa
	if(d_width != w || d_height != h)
	{
		d_width = w;
		d_height = h;

		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, d_width, 
			d_height, 0, GL_RED, GL_UNSIGNED_BYTE, data);
	}
	else
	{
		// Nadpisujemy poprzednie dane
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, d_width,
			d_height, GL_RED, GL_UNSIGNED_BYTE, data);
	}

	GLenum err = glGetError();
	if(err) printf("OpenGL Error: 0x0%x\n", err);

	//qDebug() << "         </" << __FUNCTION__ ">";
	//doneCurrent();
}

void GLDummyWidget::initialize_impl()
{
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
}

// _____________________________________________________________________________

GLWidget::GLWidget(QWidget* parent, 
	const QGLWidget* shareWidget)
	: QGLWidget(parent, shareWidget)
	, d_vboQuad(0)
	, d_surface(0)
	, d_prog(nullptr)
	, d_init(false)
{
}

GLWidget::~GLWidget()
{
	// QGLShaderProgram sie "sam" zwolni
	if(d_vboQuad)
		glDeleteBuffers(1, &d_vboQuad);
}

void GLWidget::initializeGL()
{
	if(d_init)
		return;
	d_init = true;

	GLenum err = glewInit();
	if(err != GLEW_OK)
	{
		QString msg = "GLEW error: ";
		msg += QLatin1String(
			reinterpret_cast<const char*>(glewGetErrorString(err)));

		emit error(msg);
		return;
	}
	printf("Using OpenGL %s version\n", glGetString(GL_VERSION));

	if(!GLEW_VERSION_2_0)
	{
		emit error("OpenGL 2.0 version or more is required");
		return;
	}

	// -------------------------------
	// Pare podstawowych ustawien

	glDisable(GL_DEPTH_TEST);
	glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	// -------------------------------
	// Dane wierzcholkow

	struct Vertex { float x, y, s, t; };
	Vertex vertices[3] = { 
		{ -1, -1,   0, 2 },
		{  3, -1,   2, 2 },
		{ -1,  3,   0, 0 }
	};

	printf(" * Creating vertex buffer object\n");

	glGenBuffers(1, &d_vboQuad);
	glBindBuffer(GL_ARRAY_BUFFER, d_vboQuad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Pozycja
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
		sizeof(Vertex), (GLubyte*)nullptr);
	glEnableVertexAttribArray(0);
	// Koordynaty tekstury
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 
		sizeof(Vertex), (GLubyte*)nullptr + sizeof(float)*2);
	glEnableVertexAttribArray(1);

	// -------------------------------
	// Tworzenie shaderow

	printf(" * Creating shaders\n");
	d_prog = new QGLShaderProgram(this);

	if(!d_prog->addShaderFromSourceFile
		(QGLShader::Vertex, QLatin1String("shaders/vertex.glsl")))
	{
		emit error(d_prog->log());
		return;
	}

	if(!d_prog->addShaderFromSourceFile
		(QGLShader::Fragment, QLatin1String("shaders/fragment.glsl")))
	{
		emit error(d_prog->log());
		return;
	}

	if(!d_prog->link())
	{
		emit error(d_prog->log());
		return;
	}

	if(!d_prog->bind())
	{
		emit error("Couldn't bound shader program");
		return;
	}

	d_prog->setUniformValue("surface", 0);
	d_prog->bindAttributeLocation("in_pos", 0);
	d_prog->bindAttributeLocation("in_texCoord", 1);

	err = glGetError();
	if(err) printf("OpenGL Error: 0x0%x\n", err);

	printf(" * Done initializing OpenGL\n");
	emit initialized();
}

void GLWidget::paintGL()
{
	//qDebug() << "         <" << __FUNCTION__ ">";

	glClear(GL_COLOR_BUFFER_BIT);	

	if(d_surface != 0)
	{
		// Draw full screen quad
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, d_surface);

		// Jesli tego nie wywolam to tekstura po zmianie rozmiaru sie nie odswiezy :<
		GLint w, h;
		glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
		glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);

		//qDebug() << "Texture to be drawn:" << w << h;

		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	GLenum err = glGetError();
	if(err) printf("OpenGL Error: 0x0%x\n", err);

	//qDebug() << "         </" << __FUNCTION__ ">";
}

void GLWidget::resizeGL(int width, int height)
{
	//qDebug() << "         <" << __FUNCTION__ ">";

	// no need for calling makeCurrent as it is already done by Qt
	glViewport(0, 0, width, height);

	GLenum err = glGetError();
	if(err) printf("OpenGL Error: 0x0%x\n", err);

	//qDebug() << "         </" << __FUNCTION__ ">";
}
