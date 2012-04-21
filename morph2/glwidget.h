#pragma once

#include <GL/glew.h>

#include <QGLWidget>
#include <QGLShader>
#include <QGLShaderProgram>

#include <opencv2/core/core.hpp>

class GLWidget : public QGLWidget
{
	Q_OBJECT
public:
	GLWidget(QWidget* parent);
	virtual ~GLWidget();

	void setSurface(const cv::Mat& surface);
	GLuint createEmptySurface(int w, int h);

signals:
	void initialized();
	void error(const QString& message);

protected:
	virtual void initializeGL();
	virtual void paintGL();
	virtual void resizeGL(int width, int height);

private:
	Q_DISABLE_COPY(GLWidget)

	GLuint surface;
	GLuint vboQuad;
	QGLShaderProgram* prog;

	int swidth; 
	int sheight;
	bool init;

private:
	void createSurface_impl(int w, int h, const void* data);
};
