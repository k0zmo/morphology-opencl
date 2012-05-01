QT += core gui opengl
CONFIG += debug_and_release warn_on precompile_header

TARGET = morph2
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
	oclcontext.cpp \
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
	glew.cpp

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
	oclcontext.h \
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
	previewproxy.h

FORMS += \
	sepreview.ui \
	settings.ui \
	oclpicker.ui \
	mainwindow.ui \
	mainwidget.ui

DEFINES += GLEW_STATIC
PRECOMPILED_HEADER = precompiled.h

QMAKE_CXXFLAGS += -std=c++0x -fopenmp
LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lOpenCL -fopenmp
