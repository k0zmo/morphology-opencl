QT       += core gui opengl

TARGET = morph
TEMPLATE = app
DESTDIR = ../bin32

SOURCES += main.cpp \
	mainwindow.cpp \
	morphop.cpp \
	morphocl.cpp \
	morphoclimage.cpp \
	morphoclbuffer.cpp \
	glwidget.cpp

HEADERS  += mainwindow.h \
	morphocl.h \
	morphop.h \
	morphoclimage.h \
	morphoclbuffer.h \
	glwidget.h \
	precompiled.h

FORMS    += mainwindow.ui \
	sepreview.ui

PRECOMPILED_HEADER = precompiled.h

# For gcc only
QMAKE_CXXFLAGS += -std=c++0x -fopenmp
LIBS += -lopencv_core -lopencv_imgproc -lOpenCL -fopenmp
LIBS += -L$(AMDAPPSDKROOT)/lib/x86_64
