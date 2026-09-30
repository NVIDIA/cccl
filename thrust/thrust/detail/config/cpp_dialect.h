// SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file cpp_dialect.h
 *  \brief Detect the version of the C++ standard used by the compiler.
 */

#pragma once

#include <thrust/detail/config/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config/compiler.h> // IWYU pragma: export

// Deprecated-compiler warnings live in thrust/detail/config/config.h, which
// includes this file. These must run before that header marks itself and
// its includes as system headers, or GCC/Clang silently drop them.

#define THRUST_CPP_DIALECT _CCCL_STD_VER

// C++17 dialect check:
#ifndef CCCL_IGNORE_DEPRECATED_CPP_DIALECT
#  if _CCCL_STD_VER < 2017
#    error Thrust requires at least C++17. Define CCCL_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
#  endif // _CCCL_STD_VER < 2017
#endif
