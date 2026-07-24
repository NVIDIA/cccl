// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_SORT_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_SORT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__nvtx/nvtx.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/execute.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sorter.h>
#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
_CCCL_TEMPLATE(
  class _Policy, class _CommRange, class _EnvRange, class _InputRange, class _BinaryOp = ::cuda::std::less<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND ::cuda::std::ranges::forward_range<_EnvRange>
                 _CCCL_AND ::cuda::experimental::__detail::__range_of_sized_random_access_ranges<_InputRange>)
void sort(const __result_policy_base<_Policy>& __policy,
          _CommRange&& __comms,
          _EnvRange&& __envs,
          _InputRange&& __range_of_input_ranges,
          _BinaryOp __cmp = {})
{
  using _Env = ::cuda::std::ranges::range_value_t<_EnvRange>;

  // Could use ::cuda::std::invocable here, but it is overkill (compile-time wise). We know
  // that get_stream_t is a normal CPO and normally callable.
  static_assert(::cuda::std::__is_callable_v<::cuda::get_stream_t, _Env>, "Environment must contain a stream");

  _CCCL_NVTX_RANGE_SCOPE("cuda::experimental::sort");

  using _Tp =
    ::cuda::std::ranges::range_value_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_InputRange>>>;

  if (::cuda::std::ranges::size(__comms) == 0)
  {
    // We have no inputs, so... nothing to do
    return;
  }

  ::cuda::experimental::__detail::__hss_sort::_HSSSorter<_Tp, _Env, _BinaryOp>::__execute(
    __policy,
    ::cuda::std::forward<_CommRange>(__comms),
    ::cuda::std::forward<_EnvRange>(__envs),
    ::cuda::std::forward<_InputRange>(__range_of_input_ranges),
    ::cuda::std::move(__cmp));
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_SORT_H
