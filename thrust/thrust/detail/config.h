// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file config.h
 *  \brief Defines platform configuration.
 */

#pragma once

#include <thrust/detail/config/config.h> // IWYU pragma: export

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/version.h> // IWYU pragma: export

#if !_CCCL_COMPILER(NVRTC)
#  include <cuda/__nvtx/nvtx.h>
#endif // !_CCCL_COMPILER(NVRTC)
