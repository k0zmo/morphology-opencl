#-------------------------------------------------
#
# Project created by QtCreator 2011-10-04T09:36:45
#
#-------------------------------------------------

QT       += core

TARGET = morph-cl
TEMPLATE = app
DESTDIR = ../bin32

INCLUDEPATH += ../morph
SOURCES += main.cpp \
           ../morph/morphocl.cpp \
           ../morph/morphoclbuffer.cpp \
           ../morph/morphoclimage.cpp \
           ../morph/morphop.cpp

HEADERS += ../morph/morphocl.h \
           ../morph/morphoclbuffer.h \
           ../morph/morphoclimage.h \
           ../morph/morphop.h

# For gcc only
QMAKE_CXXFLAGS += -std=c++0x -fopenmp
LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lOpenCL -fopenmp
LIBS += -L$(AMDAPPSDKROOT)/lib/x86_64
