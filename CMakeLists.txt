cmake_minimum_required (VERSION 2.8)

project(ViVid)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

message("Module path: " ${CMAKE_MODULE_PATH})

#find_package(Python)
find_package(PythonLibs 2 REQUIRED)
#message("Python Include: " ${PYTHON_INCLUDE_PATH})

find_package(PythonInterp 2 REQUIRED)

find_package(Boost COMPONENTS python REQUIRED)
message("Boost Python: " ${Boost_PYTHON_LIBRARY})

find_package(NumPy REQUIRED)
message("NumPy Include: " ${NUMPY_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
message("OpenCV Include: " ${OpenCV_INCLUDE_DIRS})

find_package(OpenCL REQUIRED)
message("OpenCL Include: " ${OPENCL_INCLUDE_DIR})
message("OpenCL Lib: " ${OPENCL_LIBRARIES})

find_package(CUDA REQUIRED)
message("CUDA Include: " ${CUDA_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)
message("OpenMP CXX FLAGS: " ${OpenMP_CXX_FLAGS})

#gcc flags
set(CMAKE_CXX_FLAGS "-fPIC ${OpenMP_CXX_FLAGS}")

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_20,code=sm_20 
    --compiler-bindir /opt/gcc-4.4/gcc)

cuda_compile(FLEXIBLE_FILTER_O src/FlexibleFilter.cu)
cuda_compile(BLOCK_HISTOGRAM_O src/BlockHistogram.cu)
cuda_compile(PAIRWISE_DISTANCE_O src/PairwiseDistance.cu)

include_directories (
    "${PROJECT_SOURCE_DIR}/src"
    ${CUDA_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_PATH}
    ${OPENCL_INCLUDE_DIR})

set(HEADER
    ${HEADER}
    src/BlockHistogram.hpp
    src/BlockHistogramWrapper.hpp
    src/PairwiseDistance.hpp
    src/PairwiseDistanceWrapper.hpp
    src/FlexibleFilter.hpp
    src/FlexibleFilterWrapper.hpp
    src/DeviceMatrix.hpp
    src/DeviceMatrixWrapper.hpp
    src/ContextOpenCL.hpp
    src/NumPyWrapper.hpp
    src/cl_exceptions.hpp
    src/exceptions.hpp
    src/OpenCLTypes.hpp
    src/OpenCLWrapper.hpp
    src/OpenCLKernels.hpp
)

set(SOURCE
    ${SOURCE}
    src/BlockHistogram.cpp
    src/BlockHistogramWrapper.cpp
    src/PairwiseDistance.cpp
    src/PairwiseDistanceWrapper.cpp
    src/FlexibleFilter.cpp
    src/FlexibleFilterWrapper.cpp
    src/DeviceMatrix.cpp
    src/DeviceMatrixWrapper.cpp
    src/ContextOpenCL.cpp
    src/NumPyWrapper.cpp
    src/cl_exceptions.cpp
    src/exceptions.cpp
    src/OpenCLWrapper.cpp
    src/OpenCLKernels.cpp
)

add_library(_vivid SHARED src/_vivid.cpp 
    ${HEADER} 
    ${SOURCE} 
    ${FLEXIBLE_FILTER_O}
    ${BLOCK_HISTOGRAM_O}
    ${PAIRWISE_DISTANCE_O}
    )
target_link_libraries(_vivid 
    ${Boost_PYTHON_LIBRARY} 
    ${PYTHON_LIBRARIES}
    ${OPENCL_LIBRARIES}
    ${CUDA_LIBRARIES})

#remove the lib_ prefix
set_target_properties(_vivid PROPERTIES PREFIX "")

