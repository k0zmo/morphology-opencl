#-------------------------------------------------
#
# Project created by QtCreator 2011-10-04T09:36:45
#
#-------------------------------------------------

TARGET = gauss
TEMPLATE = app
DESTDIR = ../bin32/old

INCLUDEPATH += ../morph
SOURCES += main.cpp

# For gcc only
QMAKE_CXXFLAGS += -std=c++0x 
LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lOpenCL
LIBS += -L$(AMDAPPSDKROOT)/lib/x86_64
