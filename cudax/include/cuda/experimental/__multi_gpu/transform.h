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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_TRANSFORM_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_TRANSFORM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_transform.cuh>

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm_utils.h>
#include <cuda/experimental/__multi_gpu/communicator.h>
#include <cuda/experimental/__multi_gpu/concepts.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
_CCCL_TEMPLATE(class _CommRange, class _EnvRange, class _InputRange, class _OutputRange, class _F)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND //
               ::cuda::std::ranges::input_range<_EnvRange> _CCCL_AND //
                 __range_of_sized_ra_ranges<_InputRange>)
void transform(_CommRange&& __comms,
               _EnvRange&& __envs,
               _InputRange&& __range_of_input_ranges,
               _OutputRange&& __range_of_output_ranges,
               _F&& __op)
{
  __validate_input_range<_InputRange>();

  for (auto&& [__comm, __env, __input_range, __output_range] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __range_of_input_ranges, __range_of_output_ranges))
  {
    const auto __num_items = ::cuda::std::ranges::size(__input_range);

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.device(),
      __num_items,
      ::cub::DeviceTransform::Transform,
      (::cuda::std::ranges::begin(__input_range),
       ::cuda::std::ranges::begin(__output_range),
       __num_items_fixed,
       __op,
       __env));
  }
}

_CCCL_TEMPLATE(class _Env, class _InputRange, class _OutputRange, class _F)
_CCCL_REQUIRES(::cuda::std::ranges::input_range<_InputRange>)
void transform(
  const communicator& __comm, const _Env& __env, _InputRange&& __input_range, _OutputRange&& __output_range, _F&& __op)
{
  transform(::cuda::std::span<const communicator, 1>{&__comm, /*__count=*/1},
            ::cuda::std::span<const _Env, 1>{::cuda::std::addressof(__env), /*__count=*/1},
            ::cuda::std::span<const _InputRange, 1>{::cuda::std::addressof(__input_range), /*__count=*/1},
            ::cuda::std::span<const _OutputRange, 1>{::cuda::std::addressof(__output_range), /*__count=*/1},
            ::cuda::std::forward<_F>(__op));
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_TRANSFORM_H
