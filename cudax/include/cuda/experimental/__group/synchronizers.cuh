//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZERS_CUH
#define _CUDA_EXPERIMENTAL___GROUP_SYNCHRONIZERS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/__memory/is_aligned.h>
#include <cuda/barrier>
#include <cuda/hierarchy>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/aligned_storage.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/span>

#include <cuda/experimental/__group/concepts.cuh>
#include <cuda/experimental/__group/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
// Requirements on synchronizers:
// - must be a class of `template <class _Unit, class _Level, class _Mapping>`
// - must be constructible from `decltype(_Mapping::map(...))` type
// - must be copyable
// - must implement `__sync(const _MappingResult&)` method, the method can be non-const
// - optionally can implement `__sync_aligned(const _MappingResult&) method (if not, __sync is used instead)

template <class _Unit, class _Level, class _Mapping>
class __syncwarp_synchronizer
{
  static_assert(::cuda::std::is_same_v<_Unit, thread_level>, "_Unit must be cuda::thread_level");

  unsigned __lane_mask_;

  template <class _Tp>
  _CCCL_DEVICE_API static constexpr bool __is_supported_count(_Tp __n) noexcept
  {
    return (::cuda::is_power_of_two(__n) || ::cuda::std::is_same_v<_Level, warp_level>) && __n <= 32;
  }

public:
  template <class _MappingResult>
  _CCCL_DEVICE_API __syncwarp_synchronizer(const _MappingResult& __mapping_result)
      : __lane_mask_{
          ((1u << __mapping_result.count()) - 1) << ((__mapping_result.group_rank() * __mapping_result.count()) % 32)}
  {
    static_assert(__group_mapping_result<_MappingResult>);
    if constexpr (_MappingResult::static_count() != ::cuda::std::dynamic_extent)
    {
      static_assert(__is_supported_count(_MappingResult::static_count()),
                    "unsupported count for __syncwarp_synchronizer");
    }
    else
    {
      _CCCL_ASSERT(__is_supported_count(__mapping_result.count()), "unsupported count for __syncwarp_synchronizer");
    }
  }

  template <class _MappingResult>
  _CCCL_DEVICE_API void __sync(const _MappingResult& __mapping_result) noexcept
  {
    if constexpr (!_MappingResult::is_always_exhaustive())
    {
      if (!__mapping_result.is_valid())
      {
        return;
      }
    }

    ::__syncwarp(__lane_mask_);
  }
};

// todo(dabayer):
// 1. make __barrier_synchronzier work with all levels
// 2. make __barrier_synchronizer work with dynamic group counts
// 3. allow users supply their own barriers in the group constructor
template <class _Unit, class _Level, class _Mapping>
class __barrier_synchronizer
{
  static_assert(::cuda::std::is_same_v<_Level, block_level>, "only block_level is currently supported");

  using _Barrier = barrier<(::cuda::std::is_same_v<_Level, grid_level>) ? thread_scope_device : thread_scope_block>;

  _Barrier* __barriers_;

  template <class _MappingResult>
  _CCCL_DEVICE_API void __init_barriers(const _MappingResult& __mapping_result) noexcept
  {
    _CCCL_ASSERT(::cuda::is_aligned(__barriers_, alignof(_Barrier)), "invalid alignment for barriers");

    if (__mapping_result.rank() == 0)
    {
      init(__barriers_ + __mapping_result.group_rank(), static_cast<::cuda::std::ptrdiff_t>(__mapping_result.count()));
    }

    // todo(dabayer): Do we want aligned or unaligned sync here?
    ::__syncthreads();
  }

public:
  using __barrier_type = _Barrier;

  template <class _MappingResult>
  _CCCL_DEVICE_API __barrier_synchronizer(const _MappingResult& __mapping_result) noexcept
  {
    static_assert(_MappingResult::static_group_count() != ::cuda::std::dynamic_extent,
                  "__barrier_synchronizer currently requires static group count");

    constexpr ::cuda::std::size_t __nbarriers = _MappingResult::static_group_count();
    using _BarrierStorage =
      ::cuda::std::aligned_storage_t<sizeof(_Barrier[__nbarriers]), alignof(_Barrier[__nbarriers])>;

    __shared__ _BarrierStorage __barrier_storage;
    __barriers_ = reinterpret_cast<_Barrier*>(::cuda::std::addressof(__barrier_storage));

    __init_barriers(__mapping_result);
  }

  template <class _MappingResult, ::cuda::std::size_t _NBarriers>
  _CCCL_DEVICE_API __barrier_synchronizer(const _MappingResult& __mapping_result,
                                          ::cuda::std::span<_Barrier, _NBarriers> __barriers) noexcept
      : __barriers_{__barriers.data()}
  {
    if constexpr (_MappingResult::static_group_count() != ::cuda::std::dynamic_extent
                  && _NBarriers != ::cuda::std::dynamic_extent)
    {
      static_assert(_MappingResult::static_group_count() == _NBarriers, "invalid number of barriers passed");
    }
    else
    {
      _CCCL_ASSERT(__mapping_result.group_count() == __barriers.size(), "invalid number of barriers passed");
    }

    __init_barriers(__mapping_result);
  }

  template <class _MappingResult>
  _CCCL_DEVICE_API void __sync(const _MappingResult& __mapping_result) noexcept
  {
    if constexpr (!_MappingResult::is_always_exhaustive())
    {
      if (!__mapping_result.is_valid())
      {
        return;
      }
    }

    __barriers_[__mapping_result.group_rank()].arrive_and_wait();
  }
};

// todo(dabayer): this is a temporary solution, implement something better
template <class _Unit, class _Level, class _Mapping, class = void>
struct __synchronizer_select
{
  using type = __barrier_synchronizer<_Unit, _Level, _Mapping>;
};

template <class _Level, ::cuda::std::size_t _Np>
struct __synchronizer_select<
  thread_level,
  _Level,
  group_by<_Np>,
  ::cuda::std::enable_if_t<::cuda::std::is_same_v<_Level, warp_level> || (::cuda::is_power_of_two(_Np) && _Np <= 32)>>
{
  using type = __syncwarp_synchronizer<thread_level, _Level, group_by<_Np>>;
};

template <class _Unit, class _Level, class _Mapping>
using __synchronizer_select_t = typename __synchronizer_select<_Unit, _Level, _Mapping>::type;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_SYNCHRONIZERS_CUH
