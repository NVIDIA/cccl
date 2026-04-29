cmake_minimum_required(VERSION 3.20)

# --------------------------------------------------------------------------
# LLVM/Clang/LLD — fetched via CPM as static libraries
# --------------------------------------------------------------------------
# CPM.cmake is at the cccl repo root: cccl/cmake/CPM.cmake
# From c/parallel/src/hostjit/ that's ../../../../cmake/CPM.cmake
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
add_library(hostjit_lib compiler.cpp config.cpp loader.cpp jit_compiler.cpp)

# CCCL_SOURCE_DIR points to the cccl repo root
# From c/parallel/src/hostjit -> c/parallel/src -> c/parallel -> c -> cccl
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH _src_dir) # c/parallel/src
cmake_path(GET _src_dir PARENT_PATH _c_parallel_dir) # c/parallel
cmake_path(GET _c_parallel_dir PARENT_PATH _c_dir) # c
cmake_path(GET _c_dir PARENT_PATH _cccl_root) # cccl

target_include_directories(
  hostjit_lib
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
  hostjit_lib
  PRIVATE
    CCCL_C_EXPERIMENTAL=1
    CCCL_SOURCE_DIR="${_cccl_root}"
    CLANG_RESOURCE_DIR="${llvm_project_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
    CLANG_HEADERS_DIR="${llvm_project_SOURCE_DIR}/clang/lib/Headers"
    HOSTJIT_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if (CUDAToolkit_FOUND)
  target_include_directories(hostjit_lib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH CUDA_TOOLKIT_ROOT_FROM_CMAKE)
  target_compile_definitions(
    hostjit_lib
    PRIVATE
      CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_ROOT_FROM_CMAKE}"
      CUDA_SDK_VERSION="${CUDAToolkit_VERSION_MAJOR}.0"
  )
endif()

# Link against LLVM/Clang/LLD
target_link_libraries(
  hostjit_lib
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
  target_link_libraries(hostjit_lib PUBLIC dl)
endif()

if (CUDAToolkit_FOUND)
  target_link_libraries(hostjit_lib PUBLIC CUDA::cuda_driver CUDA::cudart)
  # nvJitLink and nvfatbin are required at link and runtime.
  # nvptxcompiler is a transitive dep of libnvJitLink_static.
  # Prefer static variants on non-Windows; fall back to the dynamic imported target;
  # fall back further to find_library in case FindCUDAToolkit didn't create the target
  # (e.g. partial CTK installs on Ubuntu or Windows).
  foreach (_lib nvJitLink nvptxcompiler nvfatbin)
    if (NOT WIN32 AND TARGET CUDA::${_lib}_static)
      target_link_libraries(hostjit_lib PUBLIC CUDA::${_lib}_static)
    elseif (TARGET CUDA::${_lib})
      target_link_libraries(hostjit_lib PUBLIC CUDA::${_lib})
    else()
      find_library(
        _hostjit_${_lib}
        NAMES ${_lib}
        HINTS
          "${CUDAToolkit_LIBRARY_DIR}"
          "${CUDAToolkit_ROOT}/lib/x64"
          "${CUDAToolkit_ROOT}/lib64"
          "${CUDAToolkit_ROOT}/lib"
      )
      if (_hostjit_${_lib})
        message(STATUS "hostjit: linking ${_lib} from ${_hostjit_${_lib}}")
        target_link_libraries(hostjit_lib PUBLIC "${_hostjit_${_lib}}")
      else()
        message(
          FATAL_ERROR
          "hostjit requires ${_lib} but it was not found.\n"
          "  Ubuntu:  apt-get install libnvfatbin-<maj>-<min> or libnvjitlink-<maj>-<min>\n"
          "  Windows: reinstall the CUDA toolkit and ensure the nvfatbin/nvjitlink "
          "components are selected."
        )
      endif()
    endif()
  endforeach()
endif()

if (NOT MSVC)
  target_compile_options(hostjit_lib PRIVATE -fno-rtti)
endif()

set_target_properties(
  hostjit_lib
  PROPERTIES CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON
)

# On Windows with multi-config generators (Visual Studio), exclude hostjit
# targets from Debug builds — the LLVM Debug build causes stack overflows.
if (MSVC)
  set_target_properties(
    hostjit_lib
    PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD_DEBUG TRUE
  )
endif()
