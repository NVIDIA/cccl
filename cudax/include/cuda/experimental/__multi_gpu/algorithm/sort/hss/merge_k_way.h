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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_MERGE_K_WAY_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_MERGE_K_WAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_copy.cuh>
#include <cub/device/device_merge.cuh>

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/mdspan.h>
#include <cuda/std/__numeric/accumulate.h>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sorter.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <class _Tp, class _Env, class _BinaryOp>
template <class _Comm>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__merge_k_way_tree(
  const _Comm& __comm,
  const _Env& __env,
  const __resizable_buffer_type<_Tp>& __data,
  ::cuda::std::span<const ::cuda::std::size_t> __counts,
  ::cuda::std::span<const ::cuda::std::size_t> __displs,
  const _BinaryOp& __cmp,
  __resizable_buffer_type<_Tp>* __ret)
{
  auto __tmp_buf = __resizable_buffer_type<_Tp>{
    __ret->stream(),
    __ret->memory_resource(),
    __ret->size(),
    ::cuda::no_init,
    ::cuda::experimental::__detail::__sanitize_buffer_env(__env)};

  ::std::vector<::cuda::std::span<const _Tp>> __cur_level;
  ::std::vector<::cuda::std::span<const _Tp>> __next_level;

  __cur_level.reserve(__counts.size());
  for (::cuda::std::size_t __i = 0; __i < __counts.size(); ++__i)
  {
    __cur_level.push_back(__data.subspan(__displs[__i], __counts[__i]));
  }
  __next_level.reserve((__cur_level.size() + 1) / 2);

  auto* __next_level_buffer = __ret;
  bool __result_in_ret      = false;

  // Build a balanced merge tree, one level per iteration. Each level reads nodes from one
  // buffer and writes its next-level nodes contiguously to the other buffer:
  //
  //   level 0:  [A] [B] [C] [D]  [E]
  //                \ /    \ /     |
  //   level 1:    [AB]   [CD]    [E]
  //                  \   /        |
  //   level 2:      [ABCD]       [E]
  //                     \       /
  //   level 3:           [ABCDE]
  while (__cur_level.size() > 1)
  {
    ::cuda::std::size_t __next_off = 0;
    ::cuda::std::size_t __i        = 0;

    __next_level.clear();
    for (; __i + 1 < __cur_level.size(); __i += 2)
    {
      const auto __left_node  = __cur_level[__i];
      const auto __right_node = __cur_level[__i + 1];

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.logical_device(),
        CUB_NS_QUALIFIER::DeviceMerge::MergeKeys,
        __left_node.data(),
        __left_node.size(),
        __right_node.data(),
        __right_node.size(),
        __next_level_buffer->data() + __next_off,
        __cmp,
        __env);

      const auto __count = __left_node.size() + __right_node.size();

      __next_level.push_back(__next_level_buffer->subspan(__next_off, __count));
      __next_off += __count;
    }

    // We have an odd number of nodes, so we simply copy the odd node to the next level as a
    // carry. We could potentially refactor here to merge this node into the current level, but
    // I don't know if this is possible without a third temporary buffer.
    if (__i != __cur_level.size())
    {
      const auto __left_node = __cur_level[__i];
      // "Right" node would be a misnomer here, we are just copying to the equivalent slot in
      // the next level
      const auto __next_level_node = __next_level_buffer->subspan(__next_off, __left_node.size());

      const auto __input  = ::cuda::std::mdspan{__left_node.data(), __left_node.size()};
      const auto __output = ::cuda::std::mdspan{__next_level_node.data(), __next_level_node.size()};

      __CUDAX_MULTI_GPU_DISPATCH(__comm.logical_device(), CUB_NS_QUALIFIER::DeviceCopy::Copy, __input, __output, __env);

      __next_level.push_back(__next_level_node);
    }

    __cur_level.swap(__next_level);
    __result_in_ret     = __next_level_buffer == __ret;
    __next_level_buffer = __result_in_ret ? &__tmp_buf : __ret;
  }

  if (!__result_in_ret)
  {
    __ret->swap(__tmp_buf);
  }
}

// __data holds a series of sorted sequences. The offsets of the beginning of each such
// sequence is in __displs, while the count is in __counts.
template <class _Tp, class _Env, class _BinaryOp>
template <class _Comm>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__merge_k_way(
  const _Comm& __comm,
  const _Env& __env,
  const __resizable_buffer_type<_Tp>& __data,
  ::cuda::std::span<const ::cuda::std::size_t> __counts,
  ::cuda::std::span<const ::cuda::std::size_t> __displs,
  const _BinaryOp& __cmp,
  __resizable_buffer_type<_Tp>* __ret)
{
  _CCCL_VERIFY(__counts.size() > 1, "We should never get here for single-node. We should have exited earlier.");
  _CCCL_VERIFY(__counts.size() == __displs.size(), "Each sorted run must have a count and displacement");

  // __displs may be a capacity layout with gaps between the runs, so the merged size is the sum
  // of the run lengths, not the end of the last run.
  const auto __total = ::cuda::std::accumulate(__counts.begin(), __counts.end(), ::cuda::std::size_t{0});

  __ret->resize_discard(__ret->stream(), __total, ::cuda::no_init);

  // A small optimization here, if we have exactly 2 inputs, then we can write the merge into
  // __ret directly, and don't have to go through the whole tree reduction first.
  if (__counts.size() == 2)
  {
    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceMerge::MergeKeys,
      __data.data() + __displs[0],
      __counts[0],
      __data.data() + __displs[1],
      __counts[1],
      __ret->data(),
      __cmp,
      __env);
  }
  else
  {
    __merge_k_way_tree(__comm, __env, __data, __counts, __displs, __cmp, __ret);
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_MERGE_K_WAY_H
