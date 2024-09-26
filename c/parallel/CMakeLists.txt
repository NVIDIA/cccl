cmake_minimum_required(VERSION 3.30)

project(cccl.c LANGUAGES CUDA CXX)

add_library(cccl.c SHARED src/reduce.cu)
set_property(TARGET cccl.c PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET cccl.c PROPERTY CXX_STANDARD 20)
set_property(TARGET cccl.c PROPERTY CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

# TODO Use static versions of cudart, nvrtc, and nvJitLink
target_link_libraries(cccl.c PRIVATE CUDA::cudart
                                     CUDA::nvrtc
                                     CUDA::nvJitLink
                                     CUDA::cuda_driver)
target_compile_definitions(cccl.c PRIVATE NVRTC_GET_TYPE_NAME=1 CCCL_C_EXPERIMENTAL=1)
target_include_directories(cccl.c PUBLIC "include")

add_subdirectory(test)
