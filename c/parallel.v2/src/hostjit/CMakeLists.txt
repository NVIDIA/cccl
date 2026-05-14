cmake_minimum_required(VERSION 3.20)

# --------------------------------------------------------------------------
# LLVM/Clang/LLD — fetched via CPM as static libraries
# --------------------------------------------------------------------------
# CPM.cmake is at the cccl repo root: cccl/cmake/CPM.cmake
# From c/parallel.v2/src/hostjit/ that's ../../../../cmake/CPM.cmake
set(_cccl_cmake_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake")
if (EXISTS "${_cccl_cmake_dir}/CPM.cmake")
  include("${_cccl_cmake_dir}/CPM.cmake")
else()
  message(FATAL_ERROR "CPM.cmake not found at ${_cccl_cmake_dir}/CPM.cmake")
endif()

if (MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(
    FATAL_ERROR
    "hostjit does not support Debug builds on Windows. "
    "The statically-linked LLVM Debug build is too large and causes stack "
    "overflows at runtime. Use MinSizeRel, Release, or RelWithDebInfo instead."
  )
endif()

set(HOSTJIT_LLVM_VERSION "llvmorg-22.1.1" CACHE STRING "LLVM git tag to fetch")

# List options must be set before CPMAddPackage
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "X86;NVPTX" CACHE STRING "" FORCE)

CPMAddPackage(
  NAME llvm_project
  GIT_REPOSITORY https://github.com/llvm/llvm-project.git
  GIT_TAG ${HOSTJIT_LLVM_VERSION}
  GIT_SHALLOW ON
  SOURCE_SUBDIR llvm
  EXCLUDE_FROM_ALL YES
  OPTIONS
    "LLVM_BUILD_LLVM_C_DYLIB OFF"
    "LLVM_BUILD_TOOLS OFF"
    "LLVM_BUILD_UTILS OFF"
    "LLVM_BUILD_RUNTIME OFF"
    "LLVM_BUILD_RUNTIMES OFF"
    "LLVM_INCLUDE_BENCHMARKS OFF"
    "LLVM_INCLUDE_DOCS OFF"
    "LLVM_INCLUDE_EXAMPLES OFF"
    "LLVM_INCLUDE_RUNTIMES OFF"
    "LLVM_INCLUDE_TESTS OFF"
    "LLVM_INCLUDE_TOOLS ON"
    "LLVM_INCLUDE_UTILS OFF"
    "LLVM_ENABLE_ZLIB OFF"
    "LLVM_ENABLE_ZSTD OFF"
    "LLVM_ENABLE_TERMINFO OFF"
    "LLVM_ENABLE_BINDINGS OFF"
    "CLANG_BUILD_TOOLS OFF"
    "CLANG_ENABLE_ARCMT OFF"
    "CLANG_ENABLE_STATIC_ANALYZER OFF"
)

# Ensure the clang resource directory exists
file(
  MAKE_DIRECTORY "${llvm_project_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
)

# Find CUDA toolkit (may already be found by parent)
if (NOT CUDAToolkit_FOUND)
  find_package(CUDAToolkit)
endif()

# --------------------------------------------------------------------------
# hostjit library
# --------------------------------------------------------------------------
add_library(
  cccl.c.parallel.v2.hostjit_lib
  compiler.cpp
  config.cpp
  loader.cpp
  jit_compiler.cpp
  codegen/types.cpp
  codegen/iterators.cpp
  codegen/operators.cpp
  codegen/bitcode.cpp
  codegen/cub_call.cpp
)

# CCCL_SOURCE_DIR points to the cccl repo root
# From c/parallel.v2/src/hostjit -> c/parallel.v2/src -> c/parallel.v2 -> c -> cccl
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH _src_dir) # c/parallel.v2/src
cmake_path(GET _src_dir PARENT_PATH _c_parallel_dir) # c/parallel.v2
cmake_path(GET _c_parallel_dir PARENT_PATH _c_dir) # c
cmake_path(GET _c_dir PARENT_PATH _cccl_root) # cccl

target_include_directories(
  cccl.c.parallel.v2.hostjit_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${_c_parallel_dir}/include
    ${llvm_project_SOURCE_DIR}/llvm/include
    ${llvm_project_BINARY_DIR}/include
    ${llvm_project_SOURCE_DIR}/clang/include
    ${llvm_project_BINARY_DIR}/tools/clang/include
    ${llvm_project_SOURCE_DIR}/lld/include
    ${llvm_project_BINARY_DIR}/tools/lld/include
)

target_compile_definitions(
  cccl.c.parallel.v2.hostjit_lib
  PRIVATE
    CCCL_C_EXPERIMENTAL=1
    CCCL_SOURCE_DIR="${_cccl_root}"
    CLANG_RESOURCE_DIR="${llvm_project_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
    CLANG_HEADERS_DIR="${llvm_project_SOURCE_DIR}/clang/lib/Headers"
    HOSTJIT_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if (CUDAToolkit_FOUND)
  target_include_directories(
    cccl.c.parallel.v2.hostjit_lib
    PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
  )
  cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH CUDA_TOOLKIT_ROOT_FROM_CMAKE)
  target_compile_definitions(
    cccl.c.parallel.v2.hostjit_lib
    PRIVATE
      CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_ROOT_FROM_CMAKE}"
      CUDA_SDK_VERSION="${CUDAToolkit_VERSION_MAJOR}.0"
  )
endif()

# Link against LLVM/Clang/LLD
target_link_libraries(
  cccl.c.parallel.v2.hostjit_lib
  PUBLIC
    # LLVM
    LLVMCore
    LLVMSupport
    LLVMIRReader
    LLVMMC
    LLVMObject
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Desc
    LLVMX86Info
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
    LLVMLinker
    LLVMPasses
    # Clang
    clangAST
    clangBasic
    clangCodeGen
    clangDriver
    clangFrontend
    clangFrontendTool
    clangLex
    clangParse
    clangSema
    clangEdit
    clangAnalysis
    clangRewrite
    clangSerialization
    # LLD
    $<IF:$<PLATFORM_ID:Windows>,lldCOFF,lldELF>
    lldCommon
)

if (NOT WIN32)
  target_link_libraries(cccl.c.parallel.v2.hostjit_lib PUBLIC dl)
endif()

if (CUDAToolkit_FOUND)
  target_link_libraries(
    cccl.c.parallel.v2.hostjit_lib
    PUBLIC CUDA::cuda_driver CUDA::cudart
  )
  if (WIN32)
    # On Windows, static CUDA libs are built with /MT which conflicts with
    # the project's dynamic CRT (/MD). Use dynamic variants instead.
    target_link_libraries(
      cccl.c.parallel.v2.hostjit_lib
      PUBLIC CUDA::nvJitLink CUDA::nvfatbin
    )
  else()
    # Prefer static CUDA libs on Linux for self-contained binaries. If the
    # toolchain (e.g. lite/pip CUDA installs or some Docker images) only ships
    # the dynamic variants, fall back to those rather than failing configure.
    foreach (_cudalib nvJitLink nvptxcompiler nvfatbin)
      if (TARGET "CUDA::${_cudalib}_static")
        target_link_libraries(
          cccl.c.parallel.v2.hostjit_lib
          PUBLIC "CUDA::${_cudalib}_static"
        )
      elseif (TARGET "CUDA::${_cudalib}")
        target_link_libraries(
          cccl.c.parallel.v2.hostjit_lib
          PUBLIC "CUDA::${_cudalib}"
        )
      else()
        message(
          FATAL_ERROR
          "hostjit needs CUDA::${_cudalib}[_static] but neither variant was "
          "found by FindCUDAToolkit. Install the full CUDA toolkit "
          "(libnvjitlink-dev / libnvfatbin-dev or equivalent)."
        )
      endif()
    endforeach()
  endif()
endif()

if (NOT MSVC)
  target_compile_options(cccl.c.parallel.v2.hostjit_lib PRIVATE -fno-rtti)
endif()

set_target_properties(
  cccl.c.parallel.v2.hostjit_lib
  PROPERTIES CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON
)

# --------------------------------------------------------------------------
# Install clang headers into wheel (for self-sufficient packaging)
# --------------------------------------------------------------------------
# Clang CUDA headers we still use from the LLVM source tree.
# We DON'T install device_functions, math, or libdevice_declares — our local
# copies in cuda_minimal/ replace them.
set(
  _clang_cuda_headers_needed
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_math_forward_declares.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_builtin_vars.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_cmath.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_intrinsics.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_complex_builtins.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_texture_intrinsics.h"
)
install(
  FILES ${_clang_cuda_headers_needed}
  DESTINATION "cuda/cccl/headers/clang"
)

# Clang builtin C headers needed by our stubs and CUDA toolkit headers.
file(
  GLOB _clang_stddef_headers
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__stddef_*.h"
)
set(
  _clang_c_headers
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/limits.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/stddef.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/stdint.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__stddef_header_macro.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/float.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__float_header_macro.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/inttypes.h"
  ${_clang_stddef_headers}
)
install(FILES ${_clang_c_headers} DESTINATION "cuda/cccl/headers/clang")

# Hostjit's minimal CUDA runtime headers (replacements for upstream clang headers)
set(
  _hostjit_cuda_minimal_dir
  "${CMAKE_CURRENT_SOURCE_DIR}/include/hostjit/cuda_minimal"
)
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
# targets from Debug builds — the LLVM Debug build causes stack overflows.
if (MSVC)
  set_target_properties(
    cccl.c.parallel.v2.hostjit_lib
    PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD_DEBUG TRUE
  )
endif()
