cmake_minimum_required(VERSION 3.30)

# Find CUDA toolkit (may already be found by parent)
if (NOT CUDAToolkit_FOUND)
  find_package(CUDAToolkit)
endif()

# CCCL_SOURCE_DIR points to the cccl repo root
# From c/parallel.v2/src/hostjit -> c/parallel.v2/src -> c/parallel.v2 -> c -> cccl
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH _src_dir) # c/parallel.v2/src
cmake_path(GET _src_dir PARENT_PATH _c_parallel_dir) # c/parallel.v2
cmake_path(GET _c_parallel_dir PARENT_PATH _c_dir) # c
cmake_path(GET _c_dir PARENT_PATH _cccl_root) # cccl
set(_hostjit_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(LIBNVCC_CPM_CMAKE_PATH "${_cccl_root}/cmake/CPM.cmake")
set(LIBNVCC_HEADER_INSTALL_DESTINATION "cuda/cccl/headers/libnvcc")
set(LIBNVCC_CLANG_HEADER_INSTALL_DESTINATION "cuda/cccl/headers/clang")
if (CCCL_C_PARALLEL_V2_LIBRARY_OUTPUT_DIRECTORY)
  set(
    LIBNVCC_LIBRARY_OUTPUT_DIRECTORY
    "${CCCL_C_PARALLEL_V2_LIBRARY_OUTPUT_DIRECTORY}"
  )
endif()

add_subdirectory(libnvcc)

# --------------------------------------------------------------------------
# hostjit library
# --------------------------------------------------------------------------
add_library(
  cccl.c.parallel.v2.hostjit_lib
  config.cpp
  loader.cpp
  jit_compiler.cpp
  codegen/types.cpp
  codegen/iterators.cpp
  codegen/operators.cpp
  codegen/bitcode.cpp
  codegen/cub_call.cpp
)

target_include_directories(
  cccl.c.parallel.v2.hostjit_lib
  PUBLIC ${_hostjit_include_dir} ${_c_parallel_dir}/include
)

target_compile_definitions(
  cccl.c.parallel.v2.hostjit_lib
  PRIVATE
    CCCL_C_EXPERIMENTAL=1
    CCCL_SOURCE_DIR="${_cccl_root}"
    HOSTJIT_INCLUDE_DIR="${_hostjit_include_dir}"
)

if (DEFINED CLANG_HEADERS_DIR)
  target_compile_definitions(
    cccl.c.parallel.v2.hostjit_lib
    PRIVATE CLANG_HEADERS_DIR="${CLANG_HEADERS_DIR}"
  )
endif()

if (CUDAToolkit_FOUND)
  cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH CUDA_TOOLKIT_ROOT_FROM_CMAKE)
  target_compile_definitions(
    cccl.c.parallel.v2.hostjit_lib
    PRIVATE CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_ROOT_FROM_CMAKE}"
  )
endif()

if (NOT WIN32)
  target_link_libraries(cccl.c.parallel.v2.hostjit_lib PUBLIC dl)
endif()

target_link_libraries(cccl.c.parallel.v2.hostjit_lib PUBLIC libnvcc)

if (CUDAToolkit_FOUND)
  target_link_libraries(cccl.c.parallel.v2.hostjit_lib PUBLIC CUDA::cudart)
endif()

if (NOT MSVC)
  target_compile_options(cccl.c.parallel.v2.hostjit_lib PRIVATE -fno-rtti)
endif()

set_target_properties(
  cccl.c.parallel.v2.hostjit_lib
  PROPERTIES CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON
)

# Hostjit's minimal CUDA runtime headers (replacements for upstream clang headers)
set(_hostjit_cuda_minimal_dir "${_hostjit_include_dir}/hostjit/cuda_minimal")
file(GLOB _hostjit_cuda_minimal_headers "${_hostjit_cuda_minimal_dir}/*.h")
install(
  FILES ${_hostjit_cuda_minimal_headers}
  DESTINATION "cuda/cccl/headers/hostjit/cuda_minimal"
)

# Hostjit's stub headers (minimal C++ standard library stubs for device compilation)
# Use GLOB_RECURSE + DIRECTORY so subdirectory overrides (e.g. cuda/std/__cstdlib/)
# are also installed alongside the top-level stubs.
install(
  DIRECTORY "${_hostjit_cuda_minimal_dir}/stubs/"
  DESTINATION "cuda/cccl/headers/hostjit/cuda_minimal/stubs"
)

# On Windows with multi-config generators (Visual Studio), exclude hostjit
# targets that depend on libnvcc from Debug builds.
if (MSVC)
  set_target_properties(
    cccl.c.parallel.v2.hostjit_lib
    PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD_DEBUG TRUE
  )
endif()
