// SPDX-FileCopyrightText: Copyright (c) 2018, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/alignment.h>

#include <cuda/std/cstddef>

#define THRUST_MR_DEFAULT_ALIGNMENT alignof(THRUST_NS_QUALIFIER::detail::max_align_t)

#if __has_include(<memory_resource>)
#  define THRUST_MR_STD_MR_HEADER <memory_resource>
#  define THRUST_MR_STD_MR_NS     std::pmr
#elif __has_include(<experimental/memory_resource>)
#  define THRUST_MR_STD_MR_HEADER <experimental/memory_resource>
#  define THRUST_MR_STD_MR_NS     std::experimental::pmr
#endif
