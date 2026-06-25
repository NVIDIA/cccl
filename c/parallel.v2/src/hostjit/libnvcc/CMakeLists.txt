if (DEFINED LIBNVCC_CPM_CMAKE_PATH AND EXISTS "${LIBNVCC_CPM_CMAKE_PATH}")
  include("${LIBNVCC_CPM_CMAKE_PATH}")
else()
  find_file(LIBNVCC_CPM_CMAKE_PATH NAMES CPM.cmake PATHS ${CMAKE_MODULE_PATH} NO_DEFAULT_PATH)
  if (LIBNVCC_CPM_CMAKE_PATH)
    include("${LIBNVCC_CPM_CMAKE_PATH}")
  else()
    message(FATAL_ERROR "CPM.cmake not found. Set LIBNVCC_CPM_CMAKE_PATH or add it to CMAKE_MODULE_PATH.")
  endif()
endif()

if (MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(
    FATAL_ERROR
    "libnvcc does not support Debug builds on Windows. "
    "The statically-linked LLVM Debug build is too large and causes stack "
    "overflows at runtime. Use MinSizeRel, Release, or RelWithDebInfo instead."
  )
endif()

if (DEFINED HOSTJIT_LLVM_VERSION AND NOT DEFINED LIBNVCC_LLVM_VERSION)
  set(_libnvcc_llvm_version_default "${HOSTJIT_LLVM_VERSION}")
else()
  set(_libnvcc_llvm_version_default "llvmorg-22.1.1")
endif()
set(LIBNVCC_LLVM_VERSION "${_libnvcc_llvm_version_default}" CACHE STRING "LLVM git tag to fetch")

# List options must be set before CPMAddPackage
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "X86;NVPTX" CACHE STRING "" FORCE)

CPMAddPackage(
  NAME llvm_project
  GIT_REPOSITORY https://github.com/llvm/llvm-project.git
  GIT_TAG ${LIBNVCC_LLVM_VERSION}
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

add_library(
  libnvcc
  SHARED
  compiler.cpp
)

target_include_directories(
  libnvcc
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${llvm_project_SOURCE_DIR}/llvm/include
    ${llvm_project_BINARY_DIR}/include
    ${llvm_project_SOURCE_DIR}/clang/include
    ${llvm_project_BINARY_DIR}/tools/clang/include
    ${llvm_project_SOURCE_DIR}/lld/include
    ${llvm_project_BINARY_DIR}/tools/lld/include
)

target_compile_definitions(
  libnvcc
  PRIVATE
    CLANG_RESOURCE_DIR="${llvm_project_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
    CLANG_HEADERS_DIR="${llvm_project_SOURCE_DIR}/clang/lib/Headers"
    HOSTJIT_INCLUDE_DIR="${_hostjit_include_dir}"
)

# A toolkit is required because we need to call into nvfatbin/nvjitlink.
# FIXME: Statically link against those libraries individually rather than
# requiring a toolkit.
if (NOT CUDAToolkit_FOUND)
  message(FATAL_ERROR "libnvcc requires a CUDA toolkit")
endif()

target_include_directories(
  libnvcc
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)
cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH CUDA_TOOLKIT_ROOT_FROM_CMAKE)
target_compile_definitions(
  libnvcc
  PRIVATE
    CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_ROOT_FROM_CMAKE}"
    CUDA_SDK_VERSION="${CUDAToolkit_VERSION_MAJOR}.0"
)

# Link against LLVM/Clang/LLD
target_link_libraries(
  libnvcc
  PRIVATE
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

if (CUDAToolkit_FOUND)
  target_link_libraries(
    libnvcc
    PRIVATE CUDA::cuda_driver CUDA::cudart
  )
  if (WIN32)
    # On Windows, static CUDA libs are built with /MT which conflicts with
    # the project's dynamic CRT (/MD). Use dynamic variants instead.
    target_link_libraries(
      libnvcc
      PRIVATE CUDA::nvJitLink CUDA::nvfatbin
    )
  else()
    # Prefer static CUDA libs on Linux for self-contained binaries. If the
    # toolchain (e.g. lite/pip CUDA installs or some Docker images) only ships
    # the dynamic variants, fall back to those rather than failing configure.
    foreach (_cudalib nvJitLink nvptxcompiler nvfatbin)
      if (TARGET "CUDA::${_cudalib}_static")
        target_link_libraries(
          libnvcc
          PRIVATE "CUDA::${_cudalib}_static"
        )
      elseif (TARGET "CUDA::${_cudalib}")
        target_link_libraries(
          libnvcc
          PRIVATE "CUDA::${_cudalib}"
        )
      else()
        message(
          FATAL_ERROR
          "libnvcc needs CUDA::${_cudalib}[_static] but neither variant was "
          "found by FindCUDAToolkit. Install the full CUDA toolkit "
          "(libnvjitlink-dev / libnvfatbin-dev or equivalent)."
        )
      endif()
    endforeach()
  endif()
endif()

if (NOT MSVC)
  target_compile_options(libnvcc PRIVATE -fno-rtti)
endif()

set_target_properties(
  libnvcc
  PROPERTIES
    CXX_STANDARD 20
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if (LIBNVCC_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(
    libnvcc
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${LIBNVCC_LIBRARY_OUTPUT_DIRECTORY}"
      ARCHIVE_OUTPUT_DIRECTORY "${LIBNVCC_LIBRARY_OUTPUT_DIRECTORY}"
      RUNTIME_OUTPUT_DIRECTORY "${LIBNVCC_LIBRARY_OUTPUT_DIRECTORY}"
  )
endif()

# On Windows with multi-config generators (Visual Studio), exclude libnvcc from
# Debug builds — the LLVM Debug build causes stack overflows.
if (MSVC)
  set_target_properties(
    libnvcc
    PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD_DEBUG TRUE
  )
endif()

if (NOT LIBNVCC_HEADER_INSTALL_DESTINATION)
  set(LIBNVCC_HEADER_INSTALL_DESTINATION "include/libnvcc")
endif()
if (NOT LIBNVCC_CLANG_HEADER_INSTALL_DESTINATION)
  set(LIBNVCC_CLANG_HEADER_INSTALL_DESTINATION "include/libnvcc/clang")
endif()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/libnvcc/libnvcc.h" DESTINATION "${LIBNVCC_HEADER_INSTALL_DESTINATION}")

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
  DESTINATION "${LIBNVCC_CLANG_HEADER_INSTALL_DESTINATION}"
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
install(FILES ${_clang_c_headers} DESTINATION "${LIBNVCC_CLANG_HEADER_INSTALL_DESTINATION}")
