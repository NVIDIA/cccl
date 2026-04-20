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
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/span>

#include <cuda/experimental/__group/concepts.cuh>
#include <cuda/experimental/__group/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Tp, class = void>
inline constexpr bool __is_spannable = false;
template <class _Tp>
inline constexpr bool
  __is_spannable<_Tp, ::cuda::std::void_t<decltype(::cuda::std::span(::cuda::std::declval<_Tp>()))>> = true;

template <class _Span>
using _SpanElementType = typename _Span::element_type;

template <class _Barrier, ::cuda::std::size_t _Np>
class barrier_synchronizer
{
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

  template <class _Unit, class _Level, class _Mapping, class _MappingResult>
  [[nodiscard]] _CCCL_DEVICE_API __synchronizer_instance make_instance(
    const _Unit&, const _Level&, const _Mapping& __mapping, const _MappingResult& __mapping_result) const noexcept
  {
    static_assert(::cuda::std::is_same_v<_Level, block_level>, "only block_level is currently supported");

    if constexpr (_MappingResult::static_group_count() != ::cuda::std::dynamic_extent
                  && _Np != ::cuda::std::dynamic_extent)
    {
      static_assert(_MappingResult::static_group_count() <= _Np, "invalid number of barriers passed");
    }
    else
    {
      _CCCL_ASSERT(__mapping_result.group_count() <= __barriers_.size(), "invalid number of barriers passed");
    }

    // todo(dabayer): Enable this and fix initialization for threads with optional participation.
    static_assert(_MappingResult::is_always_exhaustive());
    // if constexpr (!_MappingResult::is_always_exhaustive())
    // {
    //   if (!__mapping_result.is_valid())
    //   {
    //     return;
    //   }
    // }

    if (__mapping_result.rank() == 0)
    {
      init(&__barriers_[__mapping_result.group_rank()], static_cast<::cuda::std::ptrdiff_t>(__mapping_result.count()));
    }

    // todo(dabayer): Do we want aligned or unaligned sync here?
    ::__syncthreads();
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
