// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file config.h
 *  \brief Defines platform configuration.
 */

#pragma once

// For _CCCL_IMPLICIT_SYSTEM_HEADER
#include <cuda/__cccl_config> // IWYU pragma: export

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
