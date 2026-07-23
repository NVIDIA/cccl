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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_IDEAL_RANK_FN_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_IDEAL_RANK_FN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail
{
// Maps a splitter index to its ideal global rank Ni/p. Used as the source of a
// transform_iterator. Shared between __sort_update_intervals and
// __sort_finalize_splitters.
struct __ideal_rank_fn
{
  ::cuda::std::uint64_t __N;
  ::cuda::std::uint64_t __comm_size;

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint64_t operator()(::cuda::std::uint64_t __i) const noexcept
  {
    return ((__i + 1) * __N) / __comm_size;
  }
};
} // namespace cuda::experimental::__detail

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_IDEAL_RANK_FN_H
