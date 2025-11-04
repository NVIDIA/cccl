// SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// For internal use only -- THRUST_PRAGMA_OMP is used to switch between
// different flavors of openmp pragmas. Pragmas are not emitted when OpenMP is
// not available.
//
// Usage:
//   Replace: #pragma omp parallel for
//   With   : THRUST_PRAGMA_OMP(parallel for)
//
#if defined(_NVHPC_STDPAR_OPENMP) && _NVHPC_STDPAR_OPENMP == 1
#  define THRUST_PRAGMA_OMP(directive) _CCCL_PRAGMA(omp_stdpar directive)
#elif defined(_OPENMP)
#  define THRUST_PRAGMA_OMP(directive) _CCCL_PRAGMA(omp directive)
#else
#  define THRUST_PRAGMA_OMP(directive)
#endif
