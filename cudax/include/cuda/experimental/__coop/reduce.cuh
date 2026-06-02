//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_REDUCE_CUH
#define _CUDA_EXPERIMENTAL___COOP_REDUCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/optional>

#include <cuda/experimental/group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <class _Hierarchy, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
__reduce_impl(this_thread<_Hierarchy>, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  return ::cub::ThreadReduce(__thread_data, __red_fn);
}

template <class _Hierarchy, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
__reduce_impl(this_warp<_Hierarchy> __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  using _WarpReduce = ::cub::WarpReduce<_Tp>;
  __shared__ typename _WarpReduce::TempStorage __scratch;

  const auto __result = _WarpReduce{__scratch}.Reduce(__thread_data, __red_fn);
  return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
}

template <class _Hierarchy, class _Tp, cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
__reduce_impl(this_block<_Hierarchy> __group, _Tp (&__thread_data)[_Np], _RedFn __red_fn)
{
  using _BlockExts = decltype(gpu_thread.extents(block, __group.hierarchy()));
  static_assert(_BlockExts::rank_dynamic() == 0,
                "cuda::coop::reduce requires the block level to have all static extents.");

  using _BlockReduce =
    ::cub::BlockReduce<_Tp,
                       static_cast<int>(_BlockExts::static_extent(0)),
                       ::cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                       static_cast<int>(_BlockExts::static_extent(1)),
                       static_cast<int>(_BlockExts::static_extent(2))>;
  __shared__ typename _BlockReduce::TempStorage __scratch;

  const auto __result = _BlockReduce{__scratch}.Reduce(__thread_data, __red_fn);
  return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
}

template <class _Group, class _Tp, ::cuda::std::size_t _Np, class _RedFn>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Tp>
reduce(_Group __group, _Tp (&__thread_data)[_Np], _RedFn&& __red_fn)
{
  static_assert(gpu_thread.static_count(__group) != ::cuda::std::dynamic_extent,
                "cuda::coop::reduce requires the group to have statically known size");

  if (!gpu_thread.is_part_of(__group))
  {
    return ::cuda::std::nullopt;
  }

  return ::cuda::experimental::coop::__reduce_impl(__group, __thread_data, __red_fn);
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_REDUCE_CUH
