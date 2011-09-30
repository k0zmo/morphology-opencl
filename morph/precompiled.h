#include <QtGui>
#include <QtGui/QMainWindow>
#include <QElapsedTimer>
#include <QFileDialog>
#include <QMessageBox>

#define CV_NO_BACKWARD_COMPATIBILITY

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cl/cl.hpp>
