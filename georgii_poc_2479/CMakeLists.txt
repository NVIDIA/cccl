cmake_minimum_required(VERSION 3.21)

project(CCCL_C_Parallel LANGUAGES CUDA CXX)

add_library(kernel SHARED kernel.cpp)
set_property(TARGET kernel PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET kernel PROPERTY CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

# TODO Use static versions of cudart, nvrtc, and nvJitLink
target_link_libraries(kernel PRIVATE
  CUDA::cudart
  CUDA::nvrtc
  CUDA::nvJitLink
  CUDA::cuda_driver)
