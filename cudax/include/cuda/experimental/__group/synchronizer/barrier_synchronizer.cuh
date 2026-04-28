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

template <class _Barrier, ::cuda::std::size_t _Np>
class barrier_synchronizer
{
  static_assert(__is_cuda_barrier_v<_Barrier>, "_Barrier must be cv-unqualified cuda::barrier type");

  ::cuda::std::span<_Barrier, _Np> __barriers_;

public:
  using barrier_type = _Barrier;

  struct __synchronizer_instance
  {
    template <class _MappingResult>
    _CCCL_DEVICE_API void
    do_sync(const _MappingResult& __mapping_result, const barrier_synchronizer& __synchronizer) const noexcept
    {
      __synchronizer.__barriers_[__mapping_result.group_rank()].arrive_and_wait();
    }

    template <class _MappingResult>
    _CCCL_DEVICE_API void
    do_sync_aligned(const _MappingResult& __mapping_result, const barrier_synchronizer& __synchronizer) const noexcept
    {
      __synchronizer.__barriers_[__mapping_result.group_rank()].arrive_and_wait();
    }
  };

  _CCCL_DEVICE_API barrier_synchronizer(::cuda::std::span<_Barrier, _Np> __barriers) noexcept
      : __barriers_(__barriers)
  {}

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::span<_Barrier, _Np> barriers() const noexcept
  {
    return __barriers_;
  }

  template <class _Unit, class _ParentGroup, class _Mapping, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API __synchronizer_instance make_instance(
    const _Unit&,
    const _ParentGroup& __parent,
    const _Mapping& __mapping,
    const _MappingResult& __mapping_result) const noexcept
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

    if (__mapping_result.is_valid() && __mapping_result.rank() == 0)
    {
      init(&__barriers_[__mapping_result.group_rank()], static_cast<::cuda::std::ptrdiff_t>(__mapping_result.count()));
    }

    // todo(dabayer): How we can expose making this aligned?
    __parent.sync();
    return {};
  }
};

template <class _Barrier, ::cuda::std::size_t _Np>
_CCCL_DEVICE barrier_synchronizer(::cuda::std::span<_Barrier, _Np>) -> barrier_synchronizer<_Barrier, _Np>;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_spannable<_Tp&> _CCCL_AND(!::cuda::std::__is_cuda_std_span_v<::cuda::std::remove_cv_t<_Tp>>))
_CCCL_DEVICE barrier_synchronizer(_Tp&)
  -> barrier_synchronizer<_SpanElementType<decltype(::cuda::std::span(::cuda::std::declval<_Tp&>()))>,
                          decltype(::cuda::std::span(::cuda::std::declval<_Tp&>()))::extent>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZER_BARRIER_SYNCHRONIZER_CUH
