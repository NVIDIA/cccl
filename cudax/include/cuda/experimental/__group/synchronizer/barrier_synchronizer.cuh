//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_BARRIER_SYNCHRONIZER_CUH
#define _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_BARRIER_SYNCHRONIZER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/barrier>
#include <cuda/hierarchy>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/span>

#include <cuda/experimental/__group/concepts.cuh>
#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Level>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_CONSTEVAL thread_scope __minimum_required_scope_for() noexcept
{
  if constexpr (::cuda::std::is_same_v<_Level, thread_level>)
  {
    return thread_scope_thread;
  }
  else if constexpr (::cuda::std::is_same_v<_Level, warp_level> || ::cuda::std::is_same_v<_Level, block_level>)
  {
    return thread_scope_block;
  }
  else if constexpr (::cuda::std::is_same_v<_Level, cluster_level> || ::cuda::std::is_same_v<_Level, grid_level>)
  {
    return thread_scope_device;
  }
  else
  {
    return thread_scope_system;
  }
}

template <class _Tp>
inline constexpr thread_scope __barrier_scope_v = thread_scope_system;
template <thread_scope _Sco, class _ComplFn>
inline constexpr thread_scope __barrier_scope_v<barrier<_Sco, _ComplFn>> = _Sco;

template <class _Barrier>
class __barrier_synchronizer_instance_view
{
  _Barrier* __barrier_;

public:
  _CCCL_DEVICE_API explicit __barrier_synchronizer_instance_view(_Barrier* __barrier) noexcept
      : __barrier_{__barrier}
  {}

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _MappingResult&, const _Hierarchy&) const noexcept
  {
    __barrier_->arrive_and_wait();
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void do_sync_aligned(const _MappingResult&, const _Hierarchy&) const noexcept
  {
    __barrier_->arrive_and_wait();
  }

  [[nodiscard]] _CCCL_DEVICE_API auto view() const noexcept
  {
    return *this;
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult&, const _Hierarchy&) const noexcept
  {}
};

template <class _Barrier, class _Unit>
class __barrier_synchronizer_instance
{
  _Barrier* __barrier_;

public:
  _CCCL_DEVICE_API explicit __barrier_synchronizer_instance(_Barrier* __barrier) noexcept
      : __barrier_{__barrier}
  {}

  // This synchronizer instance doesn't provide copy/move/assignment methods.
  __barrier_synchronizer_instance(const __barrier_synchronizer_instance&)            = delete;
  __barrier_synchronizer_instance(__barrier_synchronizer_instance&&)                 = delete;
  __barrier_synchronizer_instance& operator=(const __barrier_synchronizer_instance&) = delete;
  __barrier_synchronizer_instance& operator=(__barrier_synchronizer_instance&&)      = delete;

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void do_sync(const _MappingResult&, const _Hierarchy&) const noexcept
  {
    __barrier_->arrive_and_wait();
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void do_sync_aligned(const _MappingResult&, const _Hierarchy&) const noexcept
  {
    __barrier_->arrive_and_wait();
  }

  [[nodiscard]] _CCCL_DEVICE_API __barrier_synchronizer_instance_view<_Barrier> view() const noexcept
  {
    return __barrier_synchronizer_instance_view<_Barrier>{__barrier_};
  }

  template <class _MappingResult, class _Hierarchy>
  _CCCL_DEVICE_API void deinit(const _MappingResult& __mapping_result, const _Hierarchy& __hier) const noexcept
  {
    _CCCL_ASSERT(__mapping_result.is_valid(),
                 "internal error - invoking deinit() from a thread that is not part of the group");

    ::cuda::std::size_t __thread_rank_in_unit = 0u;
    if constexpr (!::cuda::std::is_same_v<_Unit, thread_level>)
    {
      __thread_rank_in_unit = gpu_thread.rank(_Unit{}, __hier);
    }

    if (__mapping_result.unit_rank() == 0 && __thread_rank_in_unit == 0)
    {
      ::cuda::std::destroy_at(__barrier_);
    }
  }
};

template <class _Barrier, ::cuda::std::size_t _Np>
class barrier_synchronizer
{
  static_assert(__is_cuda_barrier_v<_Barrier>, "_Barrier must be cv-unqualified cuda::barrier type");

  ::cuda::std::span<_Barrier, _Np> __barriers_;

public:
  using barrier_type = _Barrier;

  template <class _Unit>
  using __synchronizer_instance = __barrier_synchronizer_instance<_Barrier, _Unit>;

  _CCCL_DEVICE_API barrier_synchronizer(::cuda::std::span<_Barrier, _Np> __barriers) noexcept
      : __barriers_(__barriers)
  {}

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::span<_Barrier, _Np> barriers() const noexcept
  {
    return __barriers_;
  }

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API __synchronizer_instance<_Unit>
  make_instance(const _Unit&, const _ParentGroup& __parent, const _MappingResult& __mapping_result) const noexcept
  {
    using _Level = typename _ParentGroup::level_type;

    // todo(dabayer): Relax this condition if all units in the group are within a level that is smaller than _Level.
    static_assert(__barrier_scope_v<_Barrier> <= ::cuda::experimental::__minimum_required_scope_for<_Level>(),
                  "_Barrier's thread scope is insufficient for group synchronization in _Level");

    if constexpr (_MappingResult::static_group_count() != ::cuda::std::dynamic_extent
                  && _Np != ::cuda::std::dynamic_extent)
    {
      static_assert(_MappingResult::static_group_count() <= _Np, "invalid number of barriers passed");
    }
    else
    {
      _CCCL_ASSERT(__mapping_result.group_count() <= __barriers_.size(), "invalid number of barriers passed");
    }

    ::cuda::std::size_t __nthread_in_unit     = 1;
    ::cuda::std::size_t __thread_rank_in_unit = 0;
    if constexpr (!::cuda::std::is_same_v<thread_level, _Unit>)
    {
      __nthread_in_unit     = gpu_thread.count(_Unit{}, __parent.hierarchy());
      __thread_rank_in_unit = gpu_thread.rank(_Unit{}, __parent.hierarchy());
    }

    _Barrier* __group_barrier_ptr = nullptr;
    if (__mapping_result.is_valid())
    {
      __group_barrier_ptr = __barriers_.data() + __mapping_result.group_rank();
      if (__mapping_result.unit_rank() == 0 && __thread_rank_in_unit == 0)
      {
        init(__group_barrier_ptr,
             static_cast<::cuda::std::ptrdiff_t>(__mapping_result.unit_count() * __nthread_in_unit));
      }
    }

    // todo(dabayer): How we can expose making this aligned?
    __parent.sync();
    return __synchronizer_instance<_Unit>{__group_barrier_ptr};
  }
};

template <class _Barrier, ::cuda::std::size_t _Np>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES barrier_synchronizer(::cuda::std::span<_Barrier, _Np>)
  -> barrier_synchronizer<_Barrier, _Np>;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_spannable<_Tp&> _CCCL_AND(!::cuda::std::__is_cuda_std_span_v<::cuda::std::remove_cv_t<_Tp>>))
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES barrier_synchronizer(_Tp&)
  -> barrier_synchronizer<_SpanElementType<decltype(::cuda::std::span(::cuda::std::declval<_Tp&>()))>,
                          decltype(::cuda::std::span(::cuda::std::declval<_Tp&>()))::extent>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_BARRIER_SYNCHRONIZER_CUH
