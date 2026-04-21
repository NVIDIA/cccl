//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_CONCEPTS_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_CONCEPTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// NOLINTBEGIN(bugprone-reserved-identifier)

class communicator;

template <class _Range>
_CCCL_CONCEPT __range_of_communicators = _CCCL_REQUIRES_EXPR((_Range), )(
  // Groups usually need to be traversed twice, so cannot be just an input_range
  requires(::cuda::std::ranges::forward_range<_Range>),
  requires(::cuda::std::ranges::__container_compatible_range<_Range, communicator>));

template <class _Range>
_CCCL_CONCEPT __range_of_sized_ra_ranges = _CCCL_REQUIRES_EXPR((_Range), )(
  // The outer range of input ranges may need to be traversed more than once, so cannot be just
  // an input_range
  requires(::cuda::std::ranges::forward_range<_Range>),
  // CUB requires the underlying range to have random access (obviously required for GPUs)
  requires(::cuda::std::ranges::random_access_range<::cuda::std::ranges::range_reference_t<_Range>>),
  // CUB needs to know the number of items upfront, so the inner ranges must be sized
  requires(::cuda::std::ranges::sized_range<::cuda::std::ranges::range_reference_t<_Range>>));

template <class _Range, class _Tp>
_CCCL_CONCEPT __range_of_output_iters = _CCCL_REQUIRES_EXPR((_Range, _Tp), )(
  requires(::cuda::std::ranges::input_range<_Range>),
  requires(
    ::cuda::std::output_iterator<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_Range>>, _Tp>));

template <class _InputRange>
_CCCL_HOST_API constexpr void __validate_input_range() noexcept
{
  static_assert(!::cuda::std::ranges::__has_dangling_iterator<::cuda::std::ranges::range_reference_t<_InputRange>>,
                "Input range would produce a dangling iterator (either the range or one of its iterators stores a "
                "pointer to the parent on the stack, which cannot be dereferenced on device). The user must "
                "materialize the range into a concrete buffer before passing it to this algorithm, or use the "
                "iterator-based variant.");
}

// NOLINTEND(bugprone-reserved-identifier)
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___THREAD_GROUP_CONCEPTS_H
