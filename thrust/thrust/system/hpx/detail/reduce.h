// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file reduce.h
 *  \brief HPX implementation of reduce.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/system/hpx/detail/contiguous_iterator.h>
#include <thrust/system/hpx/detail/execution_policy.h>
#include <thrust/system/hpx/detail/function.h>

#include <hpx/parallel/algorithms/reduce.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename DerivedPolicy, typename InputIterator, typename OutputType, typename BinaryFunction>
OutputType reduce(execution_policy<DerivedPolicy>& exec [[maybe_unused]],
                  InputIterator first,
                  InputIterator last,
                  OutputType init,
                  BinaryFunction binary_op)
{
  // wrap binary_op
  hpx_wrapped_function<BinaryFunction> wrapped_binary_op{binary_op};

  if constexpr (::hpx::traits::is_forward_iterator_v<InputIterator>)
  {
    return ::hpx::reduce(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      init,
      wrapped_binary_op);
  }
  else
  {
    return ::hpx::reduce(first, last, init, wrapped_binary_op);
  }
}
} // end namespace system::hpx::detail

THRUST_NAMESPACE_END
