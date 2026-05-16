cmake_minimum_required(VERSION 3.21)

project(CCCL_C_Parallel_V2 LANGUAGES CUDA CXX C)

# Bootstrap CCCL cmake helpers when building c/parallel.v2 in isolation
# (i.e. not as a subdirectory of the CCCL super-project).
if (NOT COMMAND cccl_configure_target)
  # Repo root is two levels up from this file (c/parallel.v2 -> c -> cccl)
  get_filename_component(
    _cccl_root
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    ABSOLUTE
  )
  set(CCCL_SOURCE_DIR "${_cccl_root}" CACHE PATH "CCCL repo root" FORCE)
  set(
    CCCL_BINARY_DIR
    "${CMAKE_CURRENT_BINARY_DIR}"
    CACHE PATH
    "CCCL binary root"
    FORCE
  )
  include("${_cccl_root}/cmake/CCCLUtilities.cmake")
  include("${_cccl_root}/cmake/CCCLConfigureTarget.cmake")
  include("${_cccl_root}/cmake/CCCLGetDependencies.cmake")
  if (NOT TARGET cccl.compiler_interface)
    add_library(cccl.compiler_interface INTERFACE)
  endif()
endif()

option(CCCL_C_Parallel_V2_ENABLE_TESTING "Build cccl.c.parallel.v2 tests." OFF)

set(
  CCCL_C_PARALLEL_V2_LIBRARY_OUTPUT_DIRECTORY
  ""
  CACHE PATH
  "Override output directory for the cccl.c.parallel.v2 library"
)
mark_as_advanced(CCCL_C_PARALLEL_V2_LIBRARY_OUTPUT_DIRECTORY)

file(
  GLOB_RECURSE srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  "src/*.cu"
  "src/*.cpp"
)
# hostjit sources are built as a separate library
list(FILTER srcs EXCLUDE REGEX "^src/hostjit/")
# Editor lock/temp files
list(FILTER srcs EXCLUDE REGEX "/\\.#")

add_library(cccl.c.parallel.v2 SHARED ${srcs})
set_property(TARGET cccl.c.parallel.v2 PROPERTY POSITION_INDEPENDENT_CODE ON)
cccl_configure_target(cccl.c.parallel.v2 DIALECT 20)

if (CCCL_C_PARALLEL_V2_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(
    cccl.c.parallel.v2
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CCCL_C_PARALLEL_V2_LIBRARY_OUTPUT_DIRECTORY}"
      ARCHIVE_OUTPUT_DIRECTORY "${CCCL_C_PARALLEL_V2_LIBRARY_OUTPUT_DIRECTORY}"
      RUNTIME_OUTPUT_DIRECTORY "${CCCL_C_PARALLEL_V2_LIBRARY_OUTPUT_DIRECTORY}"
  )
endif()

cccl_get_cub()
cccl_get_cudatoolkit()
cccl_get_thrust()

add_subdirectory(src/hostjit)

set_target_properties(cccl.c.parallel.v2 PROPERTIES CUDA_RUNTIME_LIBRARY STATIC)
target_link_libraries(
  cccl.c.parallel.v2
  PRIVATE
    cccl.compiler_interface
    CUDA::cudart_static
    CUDA::nvrtc # for nvrtcGetTypeName in src/util/types.h
    CUDA::cuda_driver
    CUB::CUB
    Thrust::Thrust
    cccl.c.parallel.v2.hostjit_lib # transitively brings in nvJitLink, nvfatbin, nvptxcompiler
)

if (WIN32)
  target_link_libraries(cccl.c.parallel.v2 PRIVATE Dbghelp)
endif()

target_compile_definitions(
  cccl.c.parallel.v2
  PUBLIC CCCL_C_EXPERIMENTAL=1
  PRIVATE #
    NVRTC_GET_TYPE_NAME=1
    CUB_DISABLE_CDP=1
    CUB_DEFINE_RUNTIME_POLICIES
)
target_compile_options(
  cccl.c.parallel.v2
  PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>
)

target_include_directories(
  cccl.c.parallel.v2 #
  PUBLIC "include"
  PRIVATE "src" "src/hostjit/include"
)

if (CCCL_C_Parallel_V2_ENABLE_TESTING)
  add_subdirectory(test)
endif()
