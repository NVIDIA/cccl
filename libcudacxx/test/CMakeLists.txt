find_package(Python COMPONENTS Interpreter)
if (NOT Python_Interpreter_FOUND)
  message(
    FATAL_ERROR
    "Failed to find python interpreter, which is required for running tests and building a libcu++ static library."
  )
endif()

# Determine the host triple to avoid invoking `${CXX} -dumpmachine`.
include(GetHostTriple)
get_host_triple(LLVM_INFERRED_HOST_TRIPLE)

set(
  LLVM_HOST_TRIPLE
  "${LLVM_INFERRED_HOST_TRIPLE}"
  CACHE STRING
  "Host on which LLVM binaries will run"
)

# By default, we target the host, but this can be overridden at CMake
# invocation time.
set(
  LLVM_DEFAULT_TARGET_TRIPLE
  "${LLVM_HOST_TRIPLE}"
  CACHE STRING
  "Default target for which LLVM will generate code."
)
set(TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}")
message(STATUS "LLVM host triple: ${LLVM_HOST_TRIPLE}")
message(STATUS "LLVM default target triple: ${LLVM_DEFAULT_TARGET_TRIPLE}")

set(LIT_EXTRA_ARGS "" CACHE STRING "Use for additional options (e.g. -j12)")
find_program(LLVM_DEFAULT_EXTERNAL_LIT lit)
set(LLVM_LIT_ARGS "-sv ${LIT_EXTRA_ARGS}")

# Libcudacxx's main lit tests
add_subdirectory(libcudacxx)

add_subdirectory(cmake)

# Set appropriate warning levels for MSVC/sane
if ("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
  # CUDA 11.5 and down do not support '-use-local-env'
  if (MSVC)
    set(
      headertest_warning_levels_device
      -Xcompiler=/W4
      -Xcompiler=/WX
      -Wno-deprecated-gpu-targets
    )
    if ("${CMAKE_CUDA_COMPILER_VERSION}" GREATER_EQUAL "11.6.0")
      list(APPEND headertest_warning_levels_device --use-local-env)
    endif()
  else()
    set(
      headertest_warning_levels_device
      -Wall
      -Werror
      all-warnings
      -Wno-deprecated-gpu-targets
    )
  endif()

  if (
    CCCL_ENABLE_TILE
    AND "${CMAKE_CUDA_COMPILER_VERSION}" VERSION_LESS_EQUAL "13.2"
  )
    message(
      FATAL_ERROR
      "tile programs require NVCC 13.3 or later; found NVCC ${CMAKE_CUDA_COMPILER_VERSION}"
    )
  endif()
  # Set warnings for Clang as device compiler
elseif ("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
  set(
    headertest_warning_levels_device
    -Wall
    -Werror
    -Wno-unknown-cuda-version
    -Xclang=-fcuda-allow-variadic-functions
  )
  # If the CMAKE_CUDA_COMPILER is unknown, try to use gcc style warnings
else()
  set(headertest_warning_levels_device -Wall -Werror)
endif()

# Set raw host/device warnings
if (MSVC)
  set(headertest_warning_levels_host /W4 /WX)
else()
  set(headertest_warning_levels_host -Wall -Werror)
endif()

# Enable building the nvrtcc project if NVRTC is enabled
if (LIBCUDACXX_TEST_WITH_NVRTC)
  add_subdirectory(utils/nvidia/nvrtc)
endif()

add_subdirectory(nvtarget)

set(
  LIBCUDACXX_CODEGEN_FILECHECK_TESTS
  ""
  CACHE STRING
  "Semicolon-separated libcu++ FileCheck codegen suites to enable (all, atomics-ptx, atomics-sass, simd-ptx, simd-sass)."
)
set(
  libcudacxx_codegen_filecheck_valid_suites
  all
  atomics-ptx
  atomics-sass
  simd-ptx
  simd-sass
)
foreach (suite IN LISTS LIBCUDACXX_CODEGEN_FILECHECK_TESTS)
  if (NOT suite IN_LIST libcudacxx_codegen_filecheck_valid_suites)
    message(
      FATAL_ERROR
      "Unknown libcu++ FileCheck codegen suite '${suite}'. Expected one of: ${libcudacxx_codegen_filecheck_valid_suites}"
    )
  endif()
endforeach()

if (LIBCUDACXX_CODEGEN_FILECHECK_TESTS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CodegenTest.cmake")

  if (
    all IN_LIST LIBCUDACXX_CODEGEN_FILECHECK_TESTS
    OR atomics-ptx IN_LIST LIBCUDACXX_CODEGEN_FILECHECK_TESTS
  )
    set(libcudacxx_codegen_filecheck_atomics_ptx ON)
  endif()
  if (
    all IN_LIST LIBCUDACXX_CODEGEN_FILECHECK_TESTS
    OR atomics-sass IN_LIST LIBCUDACXX_CODEGEN_FILECHECK_TESTS
  )
    set(libcudacxx_codegen_filecheck_atomics_sass ON)
  endif()
  if (
    all IN_LIST LIBCUDACXX_CODEGEN_FILECHECK_TESTS
    OR simd-ptx IN_LIST LIBCUDACXX_CODEGEN_FILECHECK_TESTS
  )
    set(libcudacxx_codegen_filecheck_simd_ptx ON)
  endif()
  if (
    all IN_LIST LIBCUDACXX_CODEGEN_FILECHECK_TESTS
    OR simd-sass IN_LIST LIBCUDACXX_CODEGEN_FILECHECK_TESTS
  )
    set(libcudacxx_codegen_filecheck_simd_sass ON)
  endif()

  if (
    libcudacxx_codegen_filecheck_atomics_ptx
    OR libcudacxx_codegen_filecheck_atomics_sass
  )
    add_subdirectory(atomic_codegen)
  endif()
  if (
    libcudacxx_codegen_filecheck_simd_ptx
    OR libcudacxx_codegen_filecheck_simd_sass
  )
    add_subdirectory(simd_codegen)
  endif()
endif()
add_subdirectory(debugging)
