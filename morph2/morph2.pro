QT += core gui opengl
CONFIG += debug_and_release warn_on precompile_header console

DEFINES *= _CRT_SECURE_NO_WARNINGS

TARGET = morph2
CONFIG(debug, debug|release) {
	TARGET = $$join(TARGET,,,d)
}

TEMPLATE = app
DESTDIR = ../bin32

SOURCES += \
	configuration.cpp \
	controller.cpp \
	cvutils.cpp \
	elapsedtimer.cpp \
	glwidget.cpp \
	main.cpp \
	morphop.cpp \
	oclbayerfilter.cpp \
	oclfilter.cpp \
	oclmorphfilter.cpp \
	oclmorphhitmissfilter.cpp \
	procthread.cpp \
	sepreview.cpp \
	settings.cpp \
	capthread.cpp \
	oclthread.cpp \
	oclpicker.cpp \
	mainwidget.cpp \
	previewproxy.cpp \
	glew.cpp \
	minidumper.cpp \
    oclutils.cpp

HEADERS  += \
	blockingqueue.h \
	capthread.h \
	configuration.h \
	controller.h \
	cvutils.h \
	elapsedtimer.h \
	glwidget.h \
	morphop.h \
	oclbayerfilter.h \
	oclfilter.h \
	oclmorphfilter.h \
	oclmorphhitmissfilter.h \
	procthread.h \
	sepreview.h \
	settings.h \
	singleton.h \
	oclthread.h \
	oclpicker.h \
	mainwidget.h \
	previewproxy.h \
	minidumper.h \
    oclutils.h

FORMS += \
	sepreview.ui \
	settings.ui \
	oclpicker.ui \
	mainwindow.ui \
	mainwidget.ui

DEFINES += GLEW_STATIC
PRECOMPILED_HEADER = precompiled.h

unix {
	QMAKE_CXXFLAGS += -std=c++0x -fopenmp
	LIBS += -L../qt-opencl/lib
	LIBS += -lQtOpenCL -lQtOpenCLGL
	LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lOpenCL -fopenmp
	INCLUDEPATH +=  $$PWD/../qt-opencl/src/opencl $$PWD/../qt-opencl/src/openclgl
}

win32 {
	INCLUDEPATH += $$quote($$(AMDAPPSDKROOT))/include
	INCLUDEPATH += $$quote($$(OPENCVDIR))/include

	LIBS += -L$$quote($$(AMDAPPSDKROOT))/lib/x86/
	LIBS += -L$$quote($$(OPENCVDIR))/x86/vc10/lib

	CONFIG(debug, debug|release) {
		LIBS += -lopencv_core231d -lopencv_imgproc231d -lopencv_highgui231d
		LIBS += -lQtOpenCLd -lQtOpenCLGLd
	}
	CONFIG(release, debug|release) {
		LIBS += -lopencv_core231 -lopencv_imgproc231 -lopencv_highgui231
		LIBS += -lQtOpenCL -lQtOpenCLGL
	}

	LIBS += -lOpenCL
}
