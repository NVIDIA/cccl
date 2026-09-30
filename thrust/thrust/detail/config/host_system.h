// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// reserve 0 for undefined
#define THRUST_HOST_SYSTEM_CPP 1
#define THRUST_HOST_SYSTEM_OMP 2
#define THRUST_HOST_SYSTEM_TBB 3

#ifndef THRUST_HOST_SYSTEM
#  define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_CPP
#endif // THRUST_HOST_SYSTEM

#if THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_CPP
#  define __THRUST_HOST_SYSTEM_NAMESPACE cpp
#elif THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_OMP
#  define __THRUST_HOST_SYSTEM_NAMESPACE omp
#elif THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_TBB
#  define __THRUST_HOST_SYSTEM_NAMESPACE tbb
#endif

// clang-format off
#define __THRUST_HOST_SYSTEM_ROOT thrust/system/__THRUST_HOST_SYSTEM_NAMESPACE
#define __THRUST_HOST_SYSTEM_ALGORITH_HEADER_INCLUDE(filename) <__THRUST_HOST_SYSTEM_ROOT/filename>
#define __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(filename) <__THRUST_HOST_SYSTEM_ROOT/detail/filename>
// clang-format on
