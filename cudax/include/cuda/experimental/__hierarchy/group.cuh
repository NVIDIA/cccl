//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___HIERARCHY_GROUP_CUH
#define _CUDAX___HIERARCHY_GROUP_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bitmask.h>
#include <cuda/__cmath/ceil_div.h>
#include <cuda/__warp/lane_mask.h>
#include <cuda/hierarchy>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__bit/popcount.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <unsigned _Np>
struct group_by
{
  static constexpr unsigned value = _Np;
};

struct __uniform_hierarchy_group_tag
{};
struct __pred_hierarchy_group_tag
{};
struct __generic_hierarchy_group_tag
{};
template <unsigned _Np>
struct __group_by_hierarchy_group_tag
{
  static constexpr unsigned value = _Np;
};

template <class _InLevel, class _Hierarchy, class _Kind>
class thread_group;

template <class _InLevel, class _Hierarchy>
class thread_group<_InLevel, _Hierarchy, __uniform_hierarchy_group_tag>
{
  static_assert(__is_hierarchy_level_v<_InLevel>);

public:
  _CCCL_DEVICE_API thread_group(const _InLevel&, const _Hierarchy&) noexcept {}

  _CCCL_DEVICE_API void sync() noexcept
  {
    if constexpr (::cuda::std::is_same_v<_InLevel, warp_level>)
    {
      ::__syncwarp();
    }
    else if constexpr (::cuda::std::is_same_v<_InLevel, block_level>)
    {
      ::__syncthreads();
    }
    else
    {
      _CCCL_VERIFY(false, "not implemented");
    }
  }

  // todo: remove this and use unit.count(group) syntax instead
  template <class _Unit>
  [[nodiscard]] _CCCL_DEVICE_API unsigned count(const _Unit&) const noexcept
  {
    // _Unit{}.count(_InLevel{}, _Hierarchy{});
    _Unit{}.count(_InLevel{});
  }

  // todo: remove this and use unit.rank(group) syntax instead
  template <class _Unit>
  [[nodiscard]] _CCCL_DEVICE_API unsigned rank(const _Unit&) const noexcept
  {
    // _Unit{}.rank(_InLevel{}, _Hierarchy{});
    _Unit{}.rank(_InLevel{});
  }

  // todo: remove this and use group.count(in_level) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
  {
    return 1;
  }

  // todo: remove this and use group.rank(in_level) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned group_rank() const noexcept
  {
    return 0;
  }
};

template <class _Hierarchy>
class thread_group<warp_level, _Hierarchy, __pred_hierarchy_group_tag>
{
  unsigned __mask_;

public:
  _CCCL_DEVICE_API thread_group(const warp_level&, const _Hierarchy&, bool __pred) noexcept
      : __mask_{::__match_any_sync(0xffff'ffffu, __pred)}
  {}

  template <class _Fn>
  _CCCL_DEVICE_API thread_group(const warp_level& __warp, const _Hierarchy& __hier, _Fn&& __pred_fn) noexcept
      : thread_group{__warp, __hier, __pred_fn(gpu_thread.rank(warp))}
  {}

  _CCCL_DEVICE_API void sync() noexcept
  {
    ::__syncwarp(__mask_);
  }

  // todo: remove this and use unit.count(group) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned count(const thread_level&) const noexcept
  {
    return ::__reduce_add_sync(__mask_, 1);
  }

  // todo: remove this and use unit.rank(group) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned rank(const thread_level&) const noexcept
  {
    const auto __all_less_lanes = ::cuda::ptx::get_sreg_lanemask_lt();
    return static_cast<unsigned>(::cuda::std::popcount(__all_less_lanes & __mask_));
  }

  // todo: remove this and use group.count(in_level) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
  {
    return 2;
  }

  // todo: remove this and use group.rank(in_level) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned group_rank() const noexcept
  {
    return __mask_ & 1;
  }
};

template <class _Hierarchy>
class thread_group<warp_level, _Hierarchy, __generic_hierarchy_group_tag>
{
  unsigned __mask_;
  unsigned __group_rank_;
  unsigned __group_count_;

public:
  _CCCL_DEVICE_API thread_group(const warp_level&, const _Hierarchy&, unsigned __group_rank) noexcept
      : __mask_{::__match_any_sync(0xffff'ffffu, __group_rank)}
      , __group_rank_{__group_rank}
      , __group_count_{::__reduce_max_sync(0xffff'ffffu, __group_rank) + 1}
  {}

  template <class _Fn>
  _CCCL_DEVICE_API thread_group(const warp_level& __warp, const _Hierarchy& __hier, _Fn&& __pred_fn) noexcept
      : thread_group{__warp, __hier, __pred_fn(gpu_thread.rank(warp))}
  {}

  _CCCL_DEVICE_API void sync() noexcept
  {
    ::__syncwarp(__mask_);
  }

  // todo: remove this and use unit.count(group) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned count(const thread_level&) const noexcept
  {
    return ::__reduce_add_sync(__mask_, 1);
  }

  // todo: remove this and use unit.rank(group) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned rank(const thread_level&) const noexcept
  {
    const auto __all_less_lanes = ::cuda::ptx::get_sreg_lanemask_lt();
    return static_cast<unsigned>(::cuda::std::popcount(__all_less_lanes & __mask_));
  }

  // todo: remove this and use group.count(in_level) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
  {
    return __group_count_;
  }

  // todo: remove this and use group.rank(in_level) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned group_rank() const noexcept
  {
    return __group_rank_;
  }
};

template <class _Hierarchy, unsigned _Np>
class thread_group<warp_level, _Hierarchy, __group_by_hierarchy_group_tag<_Np>>
{
  unsigned __mask_;

public:
  _CCCL_DEVICE_API thread_group(const warp_level&, const _Hierarchy&, group_by<_Np>) noexcept
      : __mask_{((1u << _Np) - 1) << (gpu_thread.rank_as<unsigned>(warp) / _Np)}
  {}

  template <class _Fn>
  _CCCL_DEVICE_API thread_group(const warp_level& __warp, const _Hierarchy& __hier, _Fn&& __pred_fn) noexcept
      : thread_group{__warp, __hier, __pred_fn(gpu_thread.rank(warp))}
  {}

  _CCCL_DEVICE_API void sync() noexcept
  {
    ::__syncwarp(__mask_);
  }

  // todo: remove this and use unit.count(group) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API constexpr unsigned count(const thread_level&) const noexcept
  {
    return _Np;
  }

  // todo: remove this and use unit.rank(group) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned rank(const thread_level&) const noexcept
  {
    const auto __all_less_lanes = ::cuda::ptx::get_sreg_lanemask_lt();
    return static_cast<unsigned>(::cuda::std::popcount(__all_less_lanes & __mask_));
  }

  // todo: remove this and use group.count(in_level) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
  {
    return ::cuda::ceil_div(gpu_thread.count(warp), _Np);
  }

  // todo: remove this and use group.rank(in_level) syntax instead
  [[nodiscard]] _CCCL_DEVICE_API unsigned group_rank() const noexcept
  {
    return ::cuda::std::countr_zero(__mask_) / _Np;
  }
};

_CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
_CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE thread_group(const _InLevel&, const _Hierarchy&)
  -> thread_group<_InLevel, _Hierarchy, __uniform_hierarchy_group_tag>;

_CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
_CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE thread_group(const _InLevel&, const _Hierarchy&, bool)
  -> thread_group<_InLevel, _Hierarchy, __pred_hierarchy_group_tag>;

_CCCL_TEMPLATE(class _InLevel, class _Hierarchy)
_CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE thread_group(const _InLevel&, const _Hierarchy&, unsigned)
  -> thread_group<_InLevel, _Hierarchy, __generic_hierarchy_group_tag>;

_CCCL_TEMPLATE(class _InLevel, class _Hierarchy, class _Fn)
_CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE thread_group(const _InLevel&, const _Hierarchy&, _Fn&&)
  -> thread_group<_InLevel,
                  _Hierarchy,
                  ::cuda::std::conditional_t<::cuda::std::is_same_v<::cuda::std::invoke_result_t<_Fn&&>, bool>,
                                             __pred_hierarchy_group_tag,
                                             __generic_hierarchy_group_tag>>;

_CCCL_TEMPLATE(class _InLevel, class _Hierarchy, unsigned _Np)
_CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND __is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE thread_group(const _InLevel&, const _Hierarchy&, group_by<_Np>)
  -> thread_group<_InLevel, _Hierarchy, __group_by_hierarchy_group_tag<_Np>>;
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___HIERARCHY_GROUP_CUH
