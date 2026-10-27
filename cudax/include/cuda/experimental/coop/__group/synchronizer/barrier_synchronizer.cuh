//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_BARRIER_SYNCHRONIZER_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_BARRIER_SYNCHRONIZER_CUH

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
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <thread_scope _Scope, class _Level>
inline constexpr bool __is_sufficient_scope_for_v = false;

template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_thread, thread_level> = true;

template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_block, thread_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_block, warp_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_block, block_level> = true;

template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_cluster, thread_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_cluster, warp_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_cluster, block_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_cluster, cluster_level> = true;

template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_device, thread_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_device, warp_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_device, block_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_device, cluster_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_device, grid_level> = true;

template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_system, thread_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_system, warp_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_system, block_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_system, cluster_level> = true;
template <>
inline constexpr bool __is_sufficient_scope_for_v<thread_scope_system, grid_level> = true;

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

template <class _RangeView>
class barrier_synchronizer
{
  using _Barrier _CCCL_NODEBUG = ::cuda::std::remove_reference_t<::cuda::std::ranges::range_reference_t<_RangeView>>;

  static_assert(::cuda::std::ranges::sized_range<_RangeView>, "_RangeView must be a sized range");
  static_assert(::cuda::std::ranges::forward_range<_RangeView>, "_RangeView must be a forward range");
  static_assert(::cuda::std::ranges::borrowed_range<_RangeView>, "_RangeView must be a borrowed range");
  static_assert(__is_cuda_barrier_v<_Barrier>
                  && ::cuda::std::is_lvalue_reference_v<::cuda::std::ranges::range_reference_t<_RangeView>>,
                "_RangeView's reference type must be cv-unqualified cuda::barrier& type");

  _RangeView __barriers_;

public:
  template <class _Unit>
  using __synchronizer_instance _CCCL_NODEBUG = __barrier_synchronizer_instance<_Barrier, _Unit>;

  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_RangeView, ::cuda::std::ranges::views::all_t<_Range>>)
  _CCCL_DEVICE_API barrier_synchronizer(_Range&& __barriers) noexcept(
    noexcept(::cuda::std::ranges::views::all(::cuda::std::forward<_Range>(__barriers))))
      : __barriers_{::cuda::std::ranges::views::all(::cuda::std::forward<_Range>(__barriers))}
  {}

  template <class _Unit, class _ParentGroup, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API __synchronizer_instance<_Unit>
  make_instance(const _Unit&, const _ParentGroup& __parent, const _MappingResult& __mapping_result) noexcept
  {
    using _Level = typename _ParentGroup::level_type;

    static_assert(__is_sufficient_scope_for_v<__barrier_scope_v<_Barrier>, _Level>,
                  "_Barrier's thread scope is insufficient for group synchronization in _Level");

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
      _CCCL_ASSERT(__mapping_result.group_count() <= ::cuda::std::ranges::size(__barriers_),
                   "invalid number of barriers passed");
      __group_barrier_ptr = ::cuda::std::addressof(
        *::cuda::std::ranges::next(::cuda::std::ranges::begin(__barriers_), __mapping_result.group_rank()));

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

_CCCL_TEMPLATE(class _Range, class _Ref = ::cuda::std::ranges::range_reference_t<_Range>)
_CCCL_REQUIRES(
  ::cuda::std::ranges::viewable_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>
    _CCCL_AND ::cuda::std::ranges::forward_range<_Range> _CCCL_AND ::cuda::std::ranges::borrowed_range<_Range> _CCCL_AND
      __is_cuda_barrier_v<::cuda::std::remove_reference_t<_Ref>> _CCCL_AND ::cuda::std::is_lvalue_reference_v<_Ref>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES barrier_synchronizer(_Range&&)
  -> barrier_synchronizer<::cuda::std::ranges::views::all_t<_Range>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_SYNCHRONIZER_BARRIER_SYNCHRONIZER_CUH
