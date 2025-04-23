cmake_minimum_required(VERSION 3.21)

project(CCCL_C_Parallel LANGUAGES CUDA CXX C)

option(CCCL_C_Parallel_ENABLE_TESTING "Build cccl.c.parallel tests." OFF)
option(CCCL_C_Parallel_ENABLE_HEADER_TESTING "Build cccl.c.parallel standalone headers." OFF)

# FIXME Ideally this would be handled by presets and install rules, but for now
# consumers may override this to control the target location of cccl.c.parallel.
set(CCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY "" CACHE PATH "Override output directory for the cccl.c.parallel library")
mark_as_advanced(CCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY)

file(GLOB_RECURSE srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  "src/*.cu" "src/*.cpp"
)

add_library(cccl.c.parallel SHARED ${srcs})
set_property(TARGET cccl.c.parallel PROPERTY POSITION_INDEPENDENT_CODE ON)
cccl_configure_target(cccl.c.parallel DIALECT 20)

# Override the properties set by cccl_configure_target:
if (CCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(cccl.c.parallel PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY}"
    ARCHIVE_OUTPUT_DIRECTORY "${CCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY}"
  )
endif()

add_subdirectory(src/jit_templates)

find_package(CUDAToolkit REQUIRED)

# TODO Use static versions of cudart, nvrtc, and nvJitLink
target_link_libraries(cccl.c.parallel PRIVATE
  CUDA::cudart
  CUDA::nvrtc
  CUDA::nvJitLink
  CUDA::cuda_driver
  cccl.compiler_interface_cpp20
  cccl.c.parallel.jit_template
  CUB::CUB
  Thrust::Thrust
)
target_compile_definitions(cccl.c.parallel PUBLIC CCCL_C_EXPERIMENTAL=1 _CUB_HAS_TRANSFORM_UBLKCP=0)
target_compile_definitions(cccl.c.parallel PRIVATE NVRTC_GET_TYPE_NAME=1 CUB_DISABLE_CDP=1)
target_compile_options(cccl.c.parallel PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)

target_include_directories(cccl.c.parallel PUBLIC "include")
target_include_directories(cccl.c.parallel PRIVATE "src")

if (CCCL_C_Parallel_ENABLE_TESTING)
  add_subdirectory(test)
endif()

if (CCCL_C_Parallel_ENABLE_HEADER_TESTING)
  include(cmake/CParallelHeaderTesting.cmake)
endif()
