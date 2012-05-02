OBJS := _vivid.o
OBJS += exceptions.o
OBJS += NumPyWrapper.o
OBJS += DeviceMatrixWrapper.o DeviceMatrix.o
OBJS += PairwiseDistanceWrapper.o PairwiseDistance.o PairwiseDistance.co
OBJS += FlexibleFilterWrapper.o FlexibleFilter.o FlexibleFilter.co
OBJS += BlockHistogramWrapper.o BlockHistogram.o BlockHistogram.co
OBJS += ConvolutionWrapper.o Convolution.co
OBJDIR := release
OBJS := $(addprefix $(OBJDIR)/, $(OBJS))

#CUDA_INSTALL_PATH ?= /usr
#
#ifdef cuda-install
#	CUDA_INSTALL_PATH := $(cuda-install)
#endif

SM_VERSIONS   := 10 11 12 13 20 21

OSARCH = $(shell uname -m)

NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc --compiler-bindir /opt/gcc-4.4
CXX  := g++-4.4 -fPIC
CC   := gcc-4.4 -fPIC
LINK := g++-4.4 -fPIC

INCLUDES += -I. -I$(CUDA_INSTALL_PATH)/include

CXX_OPT := \
	-O3 \
	-funroll-loops \
	-ftree-loop-distribution \
	-fmerge-all-constants \
	-ftracer \
	-fvariable-expansion-in-unroller \
	-fvpt

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

NVCCFLAGS := -Xcompiler -fPIC -arch=compute_20

define show_status
  @echo "[$(1)]" $@
endef
SILENT:=@

# 
# Set up OpenCV
#
CFLAGS   += $(shell pkg-config opencv --cflags-only-I)
CPPFLAGS += $(shell pkg-config opencv --cflags-only-I)
LIBS += $(shell pkg-config opencv --libs)

#
# Set up CUDA
#
CPPFLAGS += $(INCLUDES)
CFLAGS += $(INCLUDES)
ifneq ($(EMU),)
  NVCCFLAGS += --device-emulation
endif
LIBS += -L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart

#
# Set up OpenMP
#
OPENMP := -fopenmp

NUMPYFLAG := -I$(shell python2 -c 'import numpy; print numpy.get_include()')
CPPFLAGS += $(shell python2.7-config --include) $(NUMPYFLAG)
LIBS += $(shell python2.7-config --libs) -lboost_python

#Make rules

$(OBJDIR)/%.o: %.cpp
	$(call show_status,c++)
	$(SILENT)$(CXX) -c -o $@ $< $(OPENMP) $(CPPFLAGS) $(CXX_OPT)

$(OBJDIR)/%.co: %.cu
	$(call show_status,cuda)
	$(SILENT)$(NVCC) -c -o $@ $< $(NVCCFLAGS)

$(OBJDIR)/_vivid.so: $(OBJDIR)/_vivid.o $(OBJS) 
	$(call show_status,link)
	$(SILENT)$(CXX) -o $@ $(OPENMP) -shared $^ $(LIBS)

test: $(OBJDIR)/_vivid.so
	../test/test_reading.py

#	$(SILENT)@$(NVCC) -M $< $(NVCCFLAGS) | sed 's#$(notdir $@)#$@#' > $(DEP)

#release/_vivid.o: _vivid.cpp
#	$(call show_status,c++)
#	$(SILENT)$(CC) -c -o $@ $< $(OPENMP) $(CPPFLAGS)
