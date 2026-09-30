// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file config.h
 *  \brief Defines platform configuration.
 */

#pragma once

// For _CCCL_IMPLICIT_SYSTEM_HEADER
#include <cuda/__cccl_config> // IWYU pragma: export

// Note: Checks must happen before marking this header and its includes as system
// headers, otherwise GCC and Clang silently drop diagnostics, including
// explicit `#pragma GCC warning` originating from system headers.

#ifndef CCCL_IGNORE_DEPRECATED_COMPILER
#  if _CCCL_COMPILER(GCC, <, 7)
_CCCL_WARNING("Thrust requires at least GCC 7.0. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.")
#  elif _CCCL_COMPILER(CLANG, <, 7)
_CCCL_WARNING("Thrust requires at least Clang 7.0. Define CCCL_IGNORE_DEPRECATED_COMPILER to suppress this message.")
#  elif _CCCL_COMPILER(MSVC, <, 19, 10)
_CCCL_WARNING("Thrust requires at least MSVC 2019(19.20 / 16.0 / 14.20). Define CCCL_IGNORE_DEPRECATED_COMPILER to "
              "suppress this message.")
#  endif
#endif // CCCL_IGNORE_DEPRECATED_COMPILER

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// NOTE: The order of these #includes matters.

#include <thrust/detail/config/compiler.h> // IWYU pragma: export
#include <thrust/detail/config/cpp_dialect.h> // IWYU pragma: export
#include <thrust/detail/config/simple_defines.h> // IWYU pragma: export
// host_system.h & device_system.h must be #included as early as possible because other config headers depend on it
#include <thrust/detail/config/host_system.h> // IWYU pragma: export

#include <thrust/detail/config/device_system.h> // IWYU pragma: export
#include <thrust/detail/config/namespace.h> // IWYU pragma: export
