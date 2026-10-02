// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
template <bool Value>
inline constexpr auto bool_constant_v = ::cuda::std::bool_constant<Value>{};

template <auto Value>
using constant_t = ::cuda::std::integral_constant<decltype(Value), Value>;

template <auto Value>
inline constexpr auto constant_v = constant_t<Value>{};
} // namespace detail
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
