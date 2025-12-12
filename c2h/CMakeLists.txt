cmake_minimum_required(VERSION 3.21)

project(C2H LANGUAGES CXX CUDA)

cccl_get_catch2()

# when CCCL is used from the CTK, C2H does not need to be rebuilt when making local changes to CCCL (faster iteration)
option(C2H_USE_CCCL_FROM_CTK "Use CCCL from the CTK in c2h." OFF)

if (C2H_USE_CCCL_FROM_CTK)
  find_package(CUDAToolkit REQUIRED)
else()
  cccl_get_cccl()
  find_package(CUDAToolkit)
endif()

set(curand_default OFF)
if (CUDA_curand_LIBRARY)
  set(curand_default ON)
endif()

option(C2H_ENABLE_CURAND "Use CUDA CURAND library in c2h." ${curand_default})

add_library(
  cccl.c2h
  STATIC
  generators.cu
  generators_gen_values.cu
  generators_uniform_offsets.cu
  generators_vector.cu
)
target_include_directories(cccl.c2h PUBLIC "${C2H_SOURCE_DIR}/include")
target_link_libraries(
  cccl.c2h
  PUBLIC #
    cccl.compiler_interface
    Catch2::Catch2
)
if (NOT C2H_USE_CCCL_FROM_CTK)
  target_link_libraries(cccl.c2h PUBLIC CCCL::CCCL)
endif()
cccl_configure_target(cccl.c2h DIALECT 17)

if (C2H_ENABLE_CURAND)
  target_link_libraries(cccl.c2h PRIVATE CUDA::curand)
  target_compile_definitions(cccl.c2h PRIVATE C2H_HAS_CURAND=1)
else()
  target_compile_definitions(cccl.c2h PRIVATE C2H_HAS_CURAND=0)
endif()

add_library(cccl.c2h.main OBJECT catch2_runner.cu catch2_runner_helper.cu)
target_link_libraries(cccl.c2h.main PUBLIC cccl.c2h)
