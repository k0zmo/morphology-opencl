QT       += core gui opengl

TARGET = morph
TEMPLATE = app
DESTDIR = ../bin32/old

SOURCES += main.cpp \
	mainwindow.cpp \
	morphoperators.cpp \
	morphocl.cpp \
	morphoclimage.cpp \
	morphoclbuffer.cpp \
	glwidget.cpp \
	cvutils.cpp \
	glew.c

HEADERS  += mainwindow.h \
	morphocl.h \
	morphoperators.h \
	morphoclimage.h \
	morphoclbuffer.h \
	glwidget.h \
	cvutils.h \
	precompiled.h

FORMS    += mainwindow.ui \
	sepreview.ui \
	settings.ui

DEFINES += GLEW_STATIC
PRECOMPILED_HEADER = precompiled.h

# For gcc only
QMAKE_CXXFLAGS += -std=c++0x -fopenmp
LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lOpenCL -fopenmp
LIBS += -L$(AMDAPPSDKROOT)/lib/x86_64
