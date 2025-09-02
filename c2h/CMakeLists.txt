cmake_minimum_required(VERSION 3.21)

project(C2H LANGUAGES CXX CUDA)

cccl_get_catch2()

find_package(CCCL CONFIG REQUIRED
  NO_DEFAULT_PATH # Only check the explicit HINTS below:
  HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cccl/"
)

find_package(CUDAToolkit)

set(curand_default OFF)
if (CUDA_curand_LIBRARY)
  set(curand_default ON)
endif()

option(C2H_ENABLE_CURAND "Use CUDA CURAND library in c2h." ${curand_default})

add_library(cccl.c2h STATIC
  generators.cu
  generators_gen_values.cu
  generators_uniform_offsets.cu
  generators_vector.cu
)
target_include_directories(cccl.c2h PUBLIC "${C2H_SOURCE_DIR}/include")
target_link_libraries(cccl.c2h PUBLIC
  CCCL::CCCL
  Catch2::Catch2
)

if (C2H_ENABLE_CURAND)
  target_link_libraries(cccl.c2h PRIVATE CUDA::curand)
  target_compile_definitions(cccl.c2h PRIVATE C2H_HAS_CURAND=1)
else()
  target_compile_definitions(cccl.c2h PRIVATE C2H_HAS_CURAND=0)
endif()

add_library(cccl.c2h.main OBJECT
  catch2_runner.cpp
  catch2_runner_helper.cu
)
target_link_libraries(cccl.c2h.main PUBLIC cccl.c2h)
