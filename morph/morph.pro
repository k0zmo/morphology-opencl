#-------------------------------------------------
#
# Project created by QtCreator 2011-10-04T09:36:45
#
#-------------------------------------------------

QT       += core gui

TARGET = morph
TEMPLATE = app


SOURCES += main.cpp \
	morph.cpp \
	morphOp.cpp

HEADERS  += morph.h \
	precompiled.h

FORMS    += morph.ui

PRECOMPILED_HEADER = precompiled.h

# For gcc only
QMAKE_CXXFLAGS += -std=c++0x
LIBS += -lopencv_core -lopencv_imgproc -lOpenCL