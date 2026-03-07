// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file transform_reduce.h
 *  \brief HPX implementation of transform_reduce.
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

#include <hpx/parallel/algorithms/transform_reduce.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx::detail
{
template <typename ExecutionPolicy,
          typename InputIterator,
          typename UnaryFunction,
          typename OutputType,
          typename BinaryFunction>
OutputType transform_reduce(
  execution_policy<ExecutionPolicy>& exec [[maybe_unused]],
  InputIterator first,
  InputIterator last,
  UnaryFunction unary_op,
  OutputType init,
  BinaryFunction binary_op)
{
  // wrap op
  hpx_wrapped_function<UnaryFunction> wrapped_unary_op{unary_op};
  hpx_wrapped_function<BinaryFunction> wrapped_binary_op{binary_op};

  if constexpr (::hpx::traits::belongs_to_iterator_traversal_v<InputIterator, ::hpx::forward_traversal_tag>)
  {
    return ::hpx::transform_reduce(
      hpx::detail::to_hpx_execution_policy(exec),
      ::thrust::try_unwrap_contiguous_iterator(first),
      ::thrust::try_unwrap_contiguous_iterator(last),
      init,
      wrapped_binary_op,
      wrapped_unary_op);
  }
  else
  {
    return ::hpx::transform_reduce(first, last, init, wrapped_binary_op, wrapped_unary_op);
  }
}
} // end namespace system::hpx::detail

THRUST_NAMESPACE_END
