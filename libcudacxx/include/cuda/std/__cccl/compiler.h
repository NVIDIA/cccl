//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_COMPILER_H
#define __CCCL_COMPILER_H

// Utility to compare version numbers. To use:
// 1) Define a macro that makes a version number from major and minor numbers, e. g.:
//    #define MYPRODUCT_MAKE_VERSION(_MAJOR, _MINOR) (_MAJOR * 100 + _MINOR)
// 2) Define a macro that you will use to compare versions, e. g.:
//    #define MYPRODUCT(...) _CCCL_VERSION_COMPARE(MYPRODUCT, MYPRODUCT_##__VA_ARGS__)
//    Signatures:
//       MYPRODUCT(_PROD)                      - is the product _PROD version non-zero?
//       MYPRODUCT(_PROD, _OP, _MAJOR)         - compare the product _PROD version to _MAJOR using operator _OP
//       MYPRODUCT(_PROD, _OP, _MAJOR, _MINOR) - compare the product _PROD version to _MAJOR._MINOR using operator _OP
#define _CCCL_VERSION_COMPARE_1(_PREFIX, _VER)              (_VER != 0)
#define _CCCL_VERSION_COMPARE_3(_PREFIX, _VER, _OP, _MAJOR) _CCCL_VERSION_COMPARE_4(_PREFIX, _VER, _OP, _MAJOR, 0)
#define _CCCL_VERSION_COMPARE_4(_PREFIX, _VER, _OP, _MAJOR, _MINOR) \
  (_CCCL_VERSION_COMPARE_1(_PREFIX, _VER) && (_VER _OP _PREFIX##MAKE_VERSION(_MAJOR, _MINOR)))
#define _CCCL_VERSION_SELECT_COUNT(_ARG1, _ARG2, _ARG3, _ARG4, _ARG5, ...) _ARG5
#define _CCCL_VERSION_SELECT2(_ARGS)                                       _CCCL_VERSION_SELECT_COUNT _ARGS
// MSVC traditonal preprocessor requires an extra level of indirection
#define _CCCL_VERSION_SELECT(...)         \
  _CCCL_VERSION_SELECT2(                  \
    (__VA_ARGS__,                         \
     _CCCL_VERSION_COMPARE_4,             \
     _CCCL_VERSION_COMPARE_3,             \
     _CCCL_VERSION_COMPARE_BAD_ARG_COUNT, \
     _CCCL_VERSION_COMPARE_1,             \
     _CCCL_VERSION_COMPARE_BAD_ARG_COUNT))
#define _CCCL_VERSION_COMPARE(_PREFIX, ...) _CCCL_VERSION_SELECT(__VA_ARGS__)(_PREFIX, __VA_ARGS__)

#define _CCCL_COMPILER_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 100 + (_MINOR))
#define _CCCL_COMPILER(...)                         _CCCL_VERSION_COMPARE(_CCCL_COMPILER_, _CCCL_COMPILER_##__VA_ARGS__)

// Determine the host compiler and its version
#if defined(__INTEL_COMPILER)
#  define _CCCL_COMPILER_ICC 1
#  ifndef CCCL_SUPPRESS_ICC_DEPRECATION_WARNING
#    warning \
      "Support for the Intel C++ Compiler Classic is deprecated and will eventually be removed. Define CCCL_SUPPRESS_ICC_DEPRECATION_WARNING to suppress this warning"
#  endif // CCCL_SUPPRESS_ICC_DEPRECATION_WARNING
#elif defined(__NVCOMPILER)
#  define _CCCL_COMPILER_NVHPC _CCCL_COMPILER_MAKE_VERSION(__NVCOMPILER_MAJOR__, __NVCOMPILER_MINOR__)
#elif defined(__clang__)
#  define _CCCL_COMPILER_CLANG _CCCL_COMPILER_MAKE_VERSION(__clang_major__, __clang_minor__)
#elif defined(__GNUC__)
#  define _CCCL_COMPILER_GCC _CCCL_COMPILER_MAKE_VERSION(__GNUC__, __GNUC_MINOR__)
#elif defined(_MSC_VER)
#  define _CCCL_COMPILER_MSVC     _CCCL_COMPILER_MAKE_VERSION(_MSC_VER / 100, _MSC_VER % 100)
#  define _CCCL_COMPILER_MSVC2017 (_CCCL_COMPILER_MSVC < _CCCL_COMPILER_MAKE_VERSION(19, 20))
#  if _CCCL_COMPILER_MSVC2017 && !defined(CCCL_SUPPRESS_MSVC2017_DEPRECATION_WARNING)
#    pragma message( \
      "Support for the Visual Studio 2017 (MSC_VER < 1920) is deprecated and will eventually be removed. Define CCCL_SUPPRESS_MSVC2017_DEPRECATION_WARNING to suppress this warning")
#  endif // CCCL_SUPPRESS_ICC_DEPRECATION_WARNING
#  define _CCCL_COMPILER_MSVC2019                               \
    (_CCCL_COMPILER_MSVC >= _CCCL_COMPILER_MAKE_VERSION(19, 20) \
     && _CCCL_COMPILER_MSVC < _CCCL_COMPILER_MAKE_VERSION(19, 30))
#  define _CCCL_COMPILER_MSVC2022                               \
    (_CCCL_COMPILER_MSVC >= _CCCL_COMPILER_MAKE_VERSION(19, 30) \
     && _CCCL_COMPILER_MSVC < _CCCL_COMPILER_MAKE_VERSION(19, 40))
#elif defined(__CUDACC_RTC__)
#  define _CCCL_COMPILER_NVRTC _CCCL_COMPILER_MAKE_VERSION(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#endif

// The CUDA compiler version shares the implementation with the C++ compiler
#define _CCCL_CUDA_COMPILER_MAKE_VERSION(_MAJOR, _MINOR) _CCCL_COMPILER_MAKE_VERSION(_MAJOR, _MINOR)
#define _CCCL_CUDA_COMPILER(...)                         _CCCL_VERSION_COMPARE(_CCCL_CUDA_COMPILER_, _CCCL_CUDA_COMPILER_##__VA_ARGS__)

// Determine the cuda compiler
#if defined(__NVCC__)
#  define _CCCL_CUDA_COMPILER_NVCC _CCCL_CUDA_COMPILER_MAKE_VERSION(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#elif defined(_NVHPC_CUDA)
#  define _CCCL_CUDA_COMPILER_NVHPC _CCCL_COMPILER_NVHPC
#elif defined(__CUDA__) && _CCCL_COMPILER(CLANG)
#  define _CCCL_CUDA_COMPILER_CLANG _CCCL_COMPILER_CLANG
#elif _CCCL_COMPILER(NVRTC)
#  define _CCCL_CUDA_COMPILER_NVRTC _CCCL_COMPILER_NVRTC
#endif

#define _CCCL_CUDACC_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 1000 + (_MINOR) * 10)

// clang-cuda does not define __CUDACC_VER_MAJOR__ and friends. They are instead retrieved from the CUDA_VERSION macro
// defined in "cuda.h". clang-cuda automatically pre-includes "__clang_cuda_runtime_wrapper.h" which includes "cuda.h"
#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(NVHPC) || _CCCL_CUDA_COMPILER(NVRTC)
#  define _CCCL_CUDACC _CCCL_CUDACC_MAKE_VERSION(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__)
#elif _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_CUDACC _CCCL_CUDACC_MAKE_VERSION(CUDA_VERSION / 1000, (CUDA_VERSION % 1000) / 10)
#endif

#define _CCCL_CUDACC_BELOW(...)    _CCCL_VERSION_COMPARE(_CCCL_CUDACC_, _CCCL_CUDACC, <, __VA_ARGS__)
#define _CCCL_CUDACC_AT_LEAST(...) _CCCL_VERSION_COMPARE(_CCCL_CUDACC_, _CCCL_CUDACC, >=, __VA_ARGS__)

#if defined(_CCCL_CUDACC)
#  define _CCCL_HAS_CUDA_COMPILER 1
#endif

// Convert parameter to string
#define _CCCL_TO_STRING2(_STR) #_STR
#define _CCCL_TO_STRING(_STR)  _CCCL_TO_STRING2(_STR)

// Define the pragma for the host compiler
#if _CCCL_COMPILER(MSVC)
#  define _CCCL_PRAGMA(_ARG) __pragma(_ARG)
#else
#  define _CCCL_PRAGMA(_ARG) _Pragma(_CCCL_TO_STRING(_ARG))
#endif // _CCCL_COMPILER(MSVC)

// Define the proper object format for NVHPC and NVRTC
#if (_CCCL_COMPILER(NVHPC) && defined(__linux__)) || _CCCL_COMPILER(NVRTC)
#  ifndef __ELF__
#    define __ELF__
#  endif // !__ELF__
#endif // _CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(NVRTC)

#if (_CCCL_CUDA_COMPILER(NVCC) && defined(__CUDA_ARCH__)) || _CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(NVRTC) \
  || _CCCL_COMPILER(CLANG)
#  define _CCCL_PRAGMA_UNROLL(_N)    _CCCL_PRAGMA(unroll _N)
#  define _CCCL_PRAGMA_UNROLL_FULL() _CCCL_PRAGMA(unroll)
#elif _CCCL_COMPILER(GCC, >=, 8)
// gcc supports only #pragma GCC unroll, but that causes problems when compiling with nvcc. So, we use #pragma unroll
// when compiling device code, and #pragma GCC unroll when compiling host code, but we need to suppress the warning
// about the unknown pragma for nvcc.
// #pragma GCC unroll does not support full unrolling, so we use the maximum value that it supports.
#  define _CCCL_PRAGMA_UNROLL(_N)    _CCCL_NV_DIAG_SUPPRESS(1675) _CCCL_PRAGMA(GCC unroll _N) _CCCL_NV_DIAG_DEFAULT(1675)
#  define _CCCL_PRAGMA_UNROLL_FULL() _CCCL_PRAGMA_UNROLL(65534)
#else // ^^^ has pragma unroll support ^^^ / vvv no pragma unroll support vvv
#  define _CCCL_PRAGMA_UNROLL(_N)
#  define _CCCL_PRAGMA_UNROLL_FULL()
#endif // ^^^ no pragma unroll support ^^^

#define _CCCL_PRAGMA_NOUNROLL() _CCCL_PRAGMA_UNROLL(1)

#endif // __CCCL_COMPILER_H
