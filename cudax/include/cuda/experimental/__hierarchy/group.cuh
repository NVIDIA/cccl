//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___HIERARCHY_GROUP_CUH
#define _CUDA_EXPERIMENTAL___HIERARCHY_GROUP_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/hierarchy>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/optional>

#include <cuda/experimental/__hierarchy/fwd.cuh>
#include <cuda/experimental/__hierarchy/grid_sync.cuh>
#include <cuda/experimental/__hierarchy/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <::cuda::std::size_t _Np>
struct group_by_t
{
  _CCCL_HIDE_FROM_ABI explicit group_by_t() = default;
};

template <::cuda::std::size_t _Np>
_CCCL_GLOBAL_CONSTANT group_by_t<_Np> group_by;

template <class _InLevel, class _Hierarchy, class _Mapping = void>
class thread_group;

template <class _Hierarchy>
class thread_group<warp_level, _Hierarchy>
{
  unsigned __grank_;
  unsigned __gcount_;
  unsigned __mask_;

public:
  using level_type = thread_level;

#if _CCCL_DEVICE_COMPILATION()
  _CCCL_TEMPLATE(class _Mapping, class _HierarchyLike)
  _CCCL_REQUIRES(::cuda::std::is_invocable_v<_Mapping, unsigned>
                   _CCCL_AND ::cuda::__is_hierarchy_v<__hierarchy_type_of<_HierarchyLike>>)
  _CCCL_DEVICE_API thread_group(const warp_level&, _Mapping&& __mapping, const _HierarchyLike& __hier) noexcept
  {
    ::cuda::std::optional __maybe_rank_in_warp(__mapping(gpu_thread.rank(warp)));
    __grank_ = __maybe_rank_in_warp.value_or(~0u);

    __gcount_ = static_cast<unsigned>(::__reduce_max_sync(0xffff'ffffu, static_cast<int>(__grank_)));
    if (__gcount_ != ~0u)
    {
      __gcount_ += 1;
    }

    __mask_ = ::__match_any_sync(0xffff'ffffu, __grank_);
    if (!__maybe_rank_in_warp.has_value())
    {
      __mask_ = 0u;
    }
  }

  _CCCL_TEMPLATE(::cuda::std::size_t _Np, class _HierarchyLike)
  _CCCL_REQUIRES(::cuda::__is_hierarchy_v<__hierarchy_type_of<_HierarchyLike>>)
  _CCCL_DEVICE_API thread_group(const warp_level&, const group_by_t<_Np>&, const _HierarchyLike& __hier) noexcept
      : __grank_{gpu_thread.rank(warp) / static_cast<unsigned>(_Np)}
      , __gcount_{::cuda::ceil_div(gpu_thread.count(warp), _Np)}
      , __mask_{(_Np == 32) ? 0xffff'ffffu : (((1u << _Np) - 1) << (__grank_ * _Np))}
  {
    static_assert(_Np > 0 && _Np < 32 && ::cuda::is_power_of_two(_Np),
                  "_Np must be greater than 0, less than or equal to 32 and a power of 2.");
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_API constexpr _Tp count_as(const _InLevel& __in_level) const noexcept
  {
    if constexpr (::cuda::std::is_same_v<_InLevel, warp_level>)
    {
      return static_cast<_Tp>(__gcount_);
    }
    else
    {
      return static_cast<_Tp>(__gcount_) * warp.count_as<_Tp>(__in_level);
    }
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_API constexpr auto count(const _InLevel& __in_level) const noexcept
  {
    return count_as<unsigned>(__in_level); // todo: return correct type
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API _Tp rank_as(const _InLevel& __in_level) const noexcept
  {
    _CCCL_ASSERT(is_part_of(gpu_thread), "Thread that is not a part of a group cannot query for the group rank.");

    if constexpr (::cuda::std::is_same_v<_InLevel, warp_level>)
    {
      return static_cast<_Tp>(__grank_);
    }
    else
    {
      return static_cast<_Tp>(__grank_) + static_cast<_Tp>(__gcount_) * warp.rank_as<_Tp>(__in_level);
    }
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API auto rank(const _InLevel& __in_level) const noexcept
  {
    return rank_as<unsigned>(__in_level); // todo: return correct type
  }
#endif // _CCCL_DEVICE_COMPILATION()

  [[nodiscard]] _CCCL_DEVICE_API bool is_part_of(const thread_level&) const noexcept
  {
    return __mask_ != 0u;
  }

  _CCCL_DEVICE_API void sync() noexcept
  {
    if (__mask_ != 0u)
    {
      ::__syncwarp(__mask_);
    }
  }
};

_CCCL_TEMPLATE(class _Mapping, class _HierarchyLike)
_CCCL_REQUIRES(::cuda::std::is_invocable_v<_Mapping, unsigned>
                 _CCCL_AND ::cuda::__is_hierarchy_v<__hierarchy_type_of<_HierarchyLike>>)
_CCCL_HOST_DEVICE thread_group(const warp_level&, _Mapping&&, const _HierarchyLike&)
  -> thread_group<warp_level, _HierarchyLike>;

_CCCL_TEMPLATE(::cuda::std::size_t _Np, class _HierarchyLike)
_CCCL_REQUIRES(::cuda::__is_hierarchy_v<__hierarchy_type_of<_HierarchyLike>>)
_CCCL_HOST_DEVICE thread_group(const warp_level&, const group_by_t<_Np>&, const _HierarchyLike&)
  -> thread_group<warp_level, _HierarchyLike>;
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_GROUP_CUH
