cmake_minimum_required(VERSION 3.30)

project(CCCL_C_Parallel LANGUAGES CUDA CXX)

option(CCCL_C_Parallel_ENABLE_TESTING "Build CUDA Experimental's tests." OFF)

add_library(cccl.c.parallel SHARED src/reduce.cu)
cccl_configure_target(cccl.c.parallel DIALECT 20)
set_property(TARGET cccl.c.parallel PROPERTY POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

# TODO Use static versions of cudart, nvrtc, and nvJitLink
target_link_libraries(cccl.c.parallel PRIVATE
  CUDA::cudart
  CUDA::nvrtc
  CUDA::nvJitLink
  CUDA::cuda_driver
  cccl.compiler_interface_cpp20
)
target_compile_definitions(cccl.c.parallel PRIVATE NVRTC_GET_TYPE_NAME=1 CCCL_C_EXPERIMENTAL=1)
target_include_directories(cccl.c.parallel PUBLIC "include")

if (CCCL_C_Parallel_ENABLE_TESTING)
  add_subdirectory(test)
endif()
