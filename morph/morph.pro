#-------------------------------------------------
#
# Project created by QtCreator 2011-10-04T09:36:45
#
#-------------------------------------------------

QT       += core gui

TARGET = morph
TEMPLATE = app
DESTDIR = ../bin32

SOURCES += main.cpp \
	mainwindow.cpp \
	morphop.cpp \
	morphocl.cpp \
	morphoclimage.cpp \
	morphoclbuffer.cpp

HEADERS  += mainwindow.h \
	morphocl.h \
	morphop.h \
	morphoclimage.h \
	morphoclbuffer.h \
	precompiled.h

FORMS    += mainwindow.ui \
	sepreview.ui

PRECOMPILED_HEADER = precompiled.h

# For gcc only
QMAKE_CXXFLAGS += -std=c++0x -fopenmp
LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lOpenCL -fopenmp
LIBS += -L$(AMDAPPSDKROOT)/lib/x86_64
