#-------------------------------------------------
#
# Project created by QtCreator 2011-10-04T09:36:45
#
#-------------------------------------------------

QT       += core gui

TARGET = morph
TEMPLATE = app


SOURCES += main.cpp \
	mainwindow.cpp \
	morphop.cpp \
	morphocl.cpp

HEADERS  += mainwindow.h \
	morphocl.h \
	morphop.h \
	precompiled.h

FORMS    += mainwindow.ui \
	sepreview.ui

PRECOMPILED_HEADER = precompiled.h

# For gcc only
QMAKE_CXXFLAGS += -std=c++0x
LIBS += -lopencv_core -lopencv_imgproc -lOpenCL
LIBS += -L$(AMDAPPSDKROOT)/lib/x86_64
