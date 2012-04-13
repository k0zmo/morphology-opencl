#pragma once

#include "GL/glew.h"
#include <QGLWidget>
#include <QGLShader>
#include <QGLShaderProgram>

#include <opencv2/core/core.hpp>

class GLWidget : public QGLWidget
{
public:
	GLWidget(QWidget* parent);
	virtual ~GLWidget();

	void setSurface(const cv::Mat& surface);
	GLuint createEmptySurface(int w, int h);

protected:
	virtual void initializeGL();
	virtual void paintGL();
	virtual void resizeGL(int width, int height);

private:
	Q_DISABLE_COPY(GLWidget)

private:
	GLuint surface;
	GLuint vboQuad;
	QGLShaderProgram* prog;

	int swidth; 
	int sheight;

private:
	void createSurface_impl(int w, int h, const void* data);
};
