#include <GL/glew.h>
#include <CL/cl.hpp>
#include <functional>

#include <QtCore>
#include <QtGui>

#include <QGLWidget>
#include <QGLShader>
#include <QGLShaderProgram>

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#if !defined(_WIN32)
#include <sys/time.h>
#else
#include <windows.h>
#endif
