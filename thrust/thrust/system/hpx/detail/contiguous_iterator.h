// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename Pointer, typename Iterator>
[[nodiscard]] constexpr Iterator rewrap_contiguous_iterator(Pointer it, Iterator base)
{
  return base + (it - ::thrust::unwrap_contiguous_iterator(base));
}

template <typename Iterator>
[[nodiscard]] constexpr Iterator rewrap_contiguous_iterator(Iterator it, Iterator /*base*/)
{
  return it;
}
} // namespace system::hpx::detail

THRUST_NAMESPACE_END
