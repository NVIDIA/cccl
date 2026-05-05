// SPDX-FileCopyrightText: Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

THRUST_NAMESPACE_BEGIN
namespace cuda_cub::detail
{
template <typename Size>
struct make_unsigned_special;

template <>
struct make_unsigned_special<int>
{
  using type = unsigned int;
};

// this is special, because CUDA's atomicAdd doesn't have an overload
// for unsigned long, for some godforsaken reason
template <>
struct make_unsigned_special<long>
{
  using type = unsigned long long;
};

template <>
struct make_unsigned_special<long long>
{
  using type = unsigned long long;
};
} // namespace cuda_cub::detail
THRUST_NAMESPACE_END
