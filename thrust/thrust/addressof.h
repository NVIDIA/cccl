// SPDX-FileCopyrightText: Copyright (c) 2018, NVIDIA Corporation
// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/addressof.h>

THRUST_NAMESPACE_BEGIN
using ::cuda::std::addressof;
THRUST_NAMESPACE_END
