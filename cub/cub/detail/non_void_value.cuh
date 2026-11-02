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

#include <cub/detail/it_traits.cuh>

#include <thrust/iterator/detail/any_assign.h>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_pointer.h>

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
template <typename It, typename FallbackT, bool = ::cuda::std::is_void_v<::cuda::std::remove_pointer_t<It>>>
struct non_void_value_impl
{
  using type = FallbackT;
};

template <typename It, typename FallbackT>
struct non_void_value_impl<It, FallbackT, false>
{
  // we consider thrust::discard_iterator's value_type (`any_assign`) as `void` as well, so users can switch from
  // cub::DiscardInputIterator to thrust::discard_iterator.
  using type = ::cuda::std::_If<::cuda::std::is_void_v<it_value_t<It>>
                                  || ::cuda::std::is_same_v<it_value_t<It>, THRUST_NS_QUALIFIER::detail::any_assign>,
                                FallbackT,
                                it_value_t<It>>;
};

/**
 * The output value type
 * type = (if IteratorT's value type is void) ?
 * ... then the FallbackT,
 * ... else the IteratorT's value type
 */
template <typename It, typename FallbackT>
using non_void_value_t = typename non_void_value_impl<It, FallbackT>::type;
} // namespace detail
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
