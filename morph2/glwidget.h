#pragma once

#include <GL/glew.h>

#include <QGLWidget>
#include <QGLShader>
#include <QGLShaderProgram>

#include <opencv2/core/core.hpp>

// Widget, ktory nigdy nie bedzie wyswietlany, trzyma on tylko teksture
// ktora jest dzielona miedzy kontrolka wlasciwa (a wiec i glownym watkiem) 
// a innymi kontrolkami/watkami (chociazby oclThread)
class GLDummyWidget : public QGLWidget
{
	Q_OBJECT
public:
	GLDummyWidget(QWidget* parent,
		const QGLWidget* shareWidget = nullptr);
	virtual ~GLDummyWidget();

	void initializeWithNewSurface(int initWidth = 1, int initHeight = 1);
	void initializeWithSharedSurface(GLuint surface);

	void setSurfaceData(const cv::Mat& surface);
	GLuint resizeSurface(int w, int h);

	GLuint surface() const { return d_surface; }

private:
	GLuint d_surface;
	int d_width;
	int d_height;
	bool deleteTexUponDestruction;

private:
	void createSurface_impl(int w, int h, const void* data);
	void initialize_impl();
};

class GLWidget : public QGLWidget
{
	Q_OBJECT
public:
	GLWidget(QWidget* parent = nullptr,
		const QGLWidget* shareWidget = nullptr);
	virtual ~GLWidget();

	void setSurface(GLuint surface) { d_surface = surface; }

signals:
	void initialized();
	void error(const QString& message);

protected:
	virtual void initializeGL();
	virtual void paintGL();
	virtual void resizeGL(int width, int height);

private:
	GLuint d_vboQuad;
	GLuint d_surface;
	QGLShaderProgram* d_prog;
	bool d_init;
};
