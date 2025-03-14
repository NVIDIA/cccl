//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DATA_MOVEMENT_LOAD_H
#define _CUDA___DATA_MOVEMENT_LOAD_H

#include <cuda/__cccl_config>

#include <cstdint>

#include "cuda/std/__internal/namespaces.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER

#  include <cuda/__barrier/aligned_size.h>
#  include <cuda/__data_movement/aligned_data.h>
#  include <cuda/__data_movement/properties.h>
#  include <cuda/__ptx/instructions/ld.h>
#  include <cuda/annotated_ptr>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__bit/bit_cast.h>
#  include <cuda/std/__bit/has_single_bit.h>
#  include <cuda/std/__cccl/unreachable.h>
#  include <cuda/std/__memory/assume_aligned.h>
#  include <cuda/std/__utility/integer_sequence.h>
#  include <cuda/std/array>
#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

/***********************************************************************************************************************
 * INTERNAL LOAD FUNCTIONS
 **********************************************************************************************************************/

#  define _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, EVICT_POLICY, PREFETCH)          \
    if constexpr ((PREFETCH) == prefetch_spatial_none)                            \
    {                                                                             \
      return _CUDA_VPTX::ld_global##LOAD_BEHAVIOR##EVICT_POLICY(__ptr);           \
    }                                                                             \
    else if constexpr ((PREFETCH) == prefetch_64B)                                \
    {                                                                             \
      return _CUDA_VPTX::ld_global##LOAD_BEHAVIOR##EVICT_POLICY##_L2_64B(__ptr);  \
    }                                                                             \
    else if constexpr ((PREFETCH) == prefetch_128B)                               \
    {                                                                             \
      return _CUDA_VPTX::ld_global##LOAD_BEHAVIOR##EVICT_POLICY##_L2_128B(__ptr); \
    }                                                                             \
    else if constexpr ((PREFETCH) == prefetch_256B)                               \
    {                                                                             \
      return _CUDA_VPTX::ld_global##LOAD_BEHAVIOR##EVICT_POLICY##_L2_256B(__ptr); \
    }

#  define _CCCL_LOAD_ADD_EVICTION_POLICY(LOAD_BEHAVIOR, EVICTION_POLICY, PREFETCH) \
    if constexpr ((EVICTION_POLICY) == eviction_none)                              \
    {                                                                              \
      _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, , PREFETCH);                          \
    }                                                                              \
    else if constexpr ((EVICTION_POLICY) == eviction_normal)                       \
    {                                                                              \
      _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, _L1_evict_normal, PREFETCH);          \
    }                                                                              \
    else if constexpr ((EVICTION_POLICY) == eviction_first)                        \
    {                                                                              \
      _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, _L1_evict_first, PREFETCH);           \
    }                                                                              \
    else if constexpr ((EVICTION_POLICY) == eviction_last)                         \
    {                                                                              \
      _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, _L1_evict_last, PREFETCH);            \
    }                                                                              \
    else if constexpr ((EVICTION_POLICY) == eviction_unchanged)                    \
    {                                                                              \
      _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, _L1_evict_unchanged, PREFETCH);       \
    }                                                                              \
    else if constexpr ((EVICTION_POLICY) == eviction_no_alloc)                     \
    {                                                                              \
      _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, _L1_no_allocate, PREFETCH);           \
    }

template <typename _Tp, _MemoryAccess _Bp, _EvictionPolicyEnum _Ep>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_sm70(
  const _Tp* __ptr, __memory_access_t<_Bp> __memory_access, __eviction_policy_t<_Ep> __eviction_policy) noexcept
{
  if constexpr (__memory_access == read_write)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __eviction_policy, prefetch_spatial_none);
  }
  else
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __eviction_policy, prefetch_spatial_none);
  }
}

template <typename _Tp, _MemoryAccess _Bp, _EvictionPolicyEnum _Ep, _PrefetchSpatialEnum _Pp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_sm75(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __eviction_policy_t<_Ep> __eviction_policy,
  __prefetch_spatial_t<_Pp> __prefetch) noexcept
{
  if constexpr (__memory_access == read_write && __prefetch == prefetch_256B)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __eviction_policy, prefetch_128B); // fallback
  }
  else if constexpr (__memory_access == read_write)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __eviction_policy, __prefetch);
  }
  else if constexpr (__memory_access == read_only && __prefetch == prefetch_256B)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __eviction_policy, prefetch_128B); // fallback
  }
  else
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __eviction_policy, __prefetch);
  }
}

template <typename _Tp, _MemoryAccess _Bp, _EvictionPolicyEnum _Ep, _PrefetchSpatialEnum _Pp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_sm80(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __eviction_policy_t<_Ep> __eviction_policy,
  __prefetch_spatial_t<_Pp> __prefetch) noexcept
{
  if constexpr (__memory_access == read_write)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __eviction_policy, __prefetch);
  }
  else
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __eviction_policy, __prefetch);
  }
}

#  undef _CCCL_LOAD_ADD_PREFETCH
#  undef _CCCL_LOAD_ADD_EVICTION_POLICY

template <typename _Tp, _MemoryAccess _Bp, _EvictionPolicyEnum _Ep, _PrefetchSpatialEnum _Pp, bool _Enabled>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_arch_dispatch(
  const _Tp* __ptr,
  [[maybe_unused]] __memory_access_t<_Bp> __memory_access,
  [[maybe_unused]] __eviction_policy_t<_Ep> __eviction_policy,
  [[maybe_unused]] __prefetch_spatial_t<_Pp> __prefetch,
  [[maybe_unused]] _CacheHint<_Enabled> __cache_hint) noexcept
{
  static_assert(sizeof(_Tp) <= 16);
#  if __cccl_ptx_isa >= 830
  // clang-format off
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (return _CUDA_VDEV::__load_sm70(__ptr, __memory_access, __eviction_policy);),
    NV_PROVIDES_SM_75, (return _CUDA_VDEV::__load_sm75(__ptr, __memory_access, __eviction_policy, __prefetch);),
    NV_PROVIDES_SM_80, (return _CUDA_VDEV::__load_sm80(__ptr, __memory_access, __eviction_policy, __prefetch);),
    NV_IS_DEVICE, (return *__ptr;)); // fallback
  // clang-format on
#  elif __cccl_ptx_isa >= 740 // __cccl_ptx_isa >= 740 && __cccl_ptx_isa < 830
  if constexpr (sizeof(_Tp) <= 8)
  {
    // clang-format off
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (return _CUDA_VDEV::__load_sm70(__ptr, __memory_access, __eviction_policy);),
      NV_PROVIDES_SM_75, (return _CUDA_VDEV::__load_sm75(__ptr, __memory_access, __eviction_policy, __prefetch);),
      NV_PROVIDES_SM_80, (return _CUDA_VDEV::__load_sm80(__ptr, __memory_access, __eviction_policy, __prefetch);),
      NV_IS_DEVICE, (return *__ptr;)); // fallback
    // clang-format on
  }
  else
  {
    return *__ptr;
  }
#  else // __cccl_ptx_isa < 740
  return *__ptr;
#  endif
  _CCCL_UNREACHABLE();
}

/***********************************************************************************************************************
 * UTILITIES
 **********************************************************************************************************************/

template <typename _Tp,
          _MemoryAccess _Bp,
          _EvictionPolicyEnum _Ep,
          _PrefetchSpatialEnum _Pp,
          bool _Enabled, //
          size_t... _Ip>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE auto __unroll_load(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __eviction_policy_t<_Ep> __eviction_policy,
  __prefetch_spatial_t<_Pp> __prefetch,
  _CacheHint<_Enabled> __cache_hint,
  _CUDA_VSTD::index_sequence<_Ip...> = {})
{
  _CUDA_VSTD::array<_Tp, sizeof...(_Ip)> __tmp;
  if constexpr (__memory_access == read_write && __eviction_policy == eviction_none
                && __prefetch == prefetch_spatial_none)
  {
    ((__tmp[_Ip] = __ptr[_Ip]), ...);
  }
  else
  {
    ((__tmp[_Ip] =
        _CUDA_VDEV::__load_arch_dispatch(__ptr + _Ip, __memory_access, __eviction_policy, __prefetch, __cache_hint)),
     ...);
  }
  return __tmp;
};

/***********************************************************************************************************************
 * INTERNAL API
 **********************************************************************************************************************/

template <typename _Tp, _MemoryAccess _Bp, _EvictionPolicyEnum _Ep, _PrefetchSpatialEnum _Pp, bool _Enable>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_element(
  const _Tp* __ptr,
  [[maybe_unused]] __memory_access_t<_Bp> __memory_access,
  [[maybe_unused]] __eviction_policy_t<_Ep> __eviction_policy,
  [[maybe_unused]] __prefetch_spatial_t<_Pp> __prefetch,
  [[maybe_unused]] _CacheHint<_Enable> __cache_hint) noexcept
{
  _CCCL_ASSERT(__ptr != nullptr, "'ptr' must not be null");
  _CCCL_ASSERT(_CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % alignof(_Tp) == 0, "'ptr' must be aligned");
  _CCCL_ASSERT(__isGlobal(__ptr), "'ptr' must point to global memory");
  auto __ptr_gmem = _CUDA_VSTD::bit_cast<const _Tp*>(__cvta_generic_to_global(__ptr));
  if constexpr (__memory_access == read_write && __eviction_policy == eviction_none
                && __prefetch == prefetch_spatial_none)
  {
    return *__ptr_gmem;
  }
  else
  {
    static_assert(_CUDA_VSTD::has_single_bit(sizeof(_Tp)) || sizeof(_Tp) % 16 == 0,
                  "'sizeof(_Tp)' must be a power of 2 or multiple of 16 with non-default properties");
    constexpr auto __access_size = _CUDA_VSTD::min(sizeof(_Tp), size_t{16});
    constexpr auto __num_unroll  = sizeof(_Tp) / __access_size;
    using __aligned_data         = _AlignedData<__access_size>;
    auto __ptr2                  = reinterpret_cast<const __aligned_data*>(__ptr_gmem);
    auto __index_seq             = _CUDA_VSTD::make_index_sequence<__num_unroll>{};
    auto __tmp =
      _CUDA_VDEV::__unroll_load(__ptr2, __memory_access, __eviction_policy, __prefetch, __cache_hint, __index_seq);
    return _CUDA_VSTD::bit_cast<_Tp>(__tmp);
  }
}

template <size_t _Np,
          typename _Tp,
          size_t _Align,
          _MemoryAccess _Bp,
          _EvictionPolicyEnum _Ep,
          _PrefetchSpatialEnum _Pp,
          bool _Enable>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, _Np> __load_array(
  const _Tp* __ptr,
  aligned_size_t<_Align>,
  __memory_access_t<_Bp> __memory_access,
  __eviction_policy_t<_Ep> __eviction_policy,
  __prefetch_spatial_t<_Pp> __prefetch,
  _CacheHint<_Enable> __cache_hint) noexcept
{
  static_assert(_CUDA_VSTD::has_single_bit(_Align), "_Align must be a power of 2");
  static_assert(_Np > 0);
  static_assert(_Align >= alignof(_Tp), "_Align must be greater than or equal to alignof(_Tp)");
  static_assert(sizeof(_Tp) * _Np % _Align == 0, "Np * sizeof(_Tp) must be a multiple of _Align");
  _CCCL_ASSERT(__ptr != nullptr, "'ptr' must not be null");
  _CCCL_ASSERT(_CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % _Align == 0, "'ptr' must be aligned");
  _CCCL_ASSERT(__isGlobal(__ptr), "'ptr' must point to global memory");
  constexpr bool __is_default_access =
    __memory_access == read_write && __eviction_policy == eviction_none && __prefetch == prefetch_spatial_none;
  constexpr auto __max_align = __is_default_access ? _Align : _CUDA_VSTD::min(_Align, size_t{16});
  constexpr auto __count     = (sizeof(_Tp) * _Np) / __max_align;
  using __aligned_data       = _AlignedData<__max_align>;
  auto __ptr_gmem            = _CUDA_VSTD::bit_cast<const __aligned_data*>(__cvta_generic_to_global(__ptr));
  auto __index_seq           = _CUDA_VSTD::make_index_sequence<__count>{};
  auto __tmp =
    _CUDA_VDEV::__unroll_load(__ptr_gmem, __memory_access, __eviction_policy, __prefetch, __cache_hint, __index_seq);
  using __result_t = _CUDA_VSTD::array<_Tp, _Np>;
  return _CUDA_VSTD::bit_cast<__result_t>(__tmp);
}

/***********************************************************************************************************************
 * USER API
 **********************************************************************************************************************/

template <typename _Tp,
          _MemoryAccess _Bp        = _MemoryAccess::_ReadWrite,
          _EvictionPolicyEnum _Ep  = _EvictionPolicyEnum::_None,
          _PrefetchSpatialEnum _Pp = _PrefetchSpatialEnum::_None>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp
load(const _Tp* __ptr,
     __memory_access_t<_Bp> __memory_access     = read_write,
     __eviction_policy_t<_Ep> __eviction_policy = eviction_none,
     __prefetch_spatial_t<_Pp> __prefetch       = prefetch_spatial_none) noexcept
{
  return _CUDA_VDEV::__load_element(__ptr, __memory_access, __eviction_policy, __prefetch, __no_cache_hint);
}

template <typename _Tp,
          typename _Prop,
          _MemoryAccess _Bp        = _MemoryAccess::_ReadWrite,
          _EvictionPolicyEnum _Ep  = _EvictionPolicyEnum::_None,
          _PrefetchSpatialEnum _Pp = _PrefetchSpatialEnum::_None>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp
load(annotated_ptr<_Tp, _Prop> __ptr,
     __memory_access_t<_Bp> __memory_access     = read_write,
     __eviction_policy_t<_Ep> __eviction_policy = eviction_none,
     __prefetch_spatial_t<_Pp> __prefetch       = prefetch_spatial_none) noexcept
{
  auto __cache_hint = _CacheHint<true>{static_cast<uint64_t>(__ptr.__property())};
  return _CUDA_VDEV::__load_element(__ptr.get(), __memory_access, __eviction_policy, __prefetch, __cache_hint);
}

template <size_t _Np,
          typename _Tp,
          size_t _Align            = alignof(_Tp),
          _MemoryAccess _Bp        = _MemoryAccess::_ReadWrite,
          _EvictionPolicyEnum _Ep  = _EvictionPolicyEnum::_None,
          _PrefetchSpatialEnum _Pp = _PrefetchSpatialEnum::_None>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, _Np>
load(const _Tp* __ptr,
     aligned_size_t<_Align> __align             = aligned_size_t<_Align>{alignof(_Tp)},
     __memory_access_t<_Bp> __memory_access     = read_write,
     __eviction_policy_t<_Ep> __eviction_policy = eviction_none,
     __prefetch_spatial_t<_Pp> __prefetch       = prefetch_spatial_none) noexcept
{
  return _CUDA_VDEV::__load_array<_Np>(__ptr, __align, __memory_access, __eviction_policy, __prefetch, __no_cache_hint);
}

template <size_t _Np,
          typename _Tp,
          typename _Prop,
          size_t _Align            = alignof(_Tp),
          _MemoryAccess _Bp        = _MemoryAccess::_ReadWrite,
          _EvictionPolicyEnum _Ep  = _EvictionPolicyEnum::_None,
          _PrefetchSpatialEnum _Pp = _PrefetchSpatialEnum::_None>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, _Np>
load(annotated_ptr<_Tp, _Prop> __ptr,
     aligned_size_t<_Align> __align             = aligned_size_t<_Align>{alignof(_Tp)},
     __memory_access_t<_Bp> __memory_access     = read_write,
     __eviction_policy_t<_Ep> __eviction_policy = eviction_none,
     __prefetch_spatial_t<_Pp> __prefetch       = prefetch_spatial_none) noexcept
{
  auto __cache_hint = _CacheHint<true>{static_cast<uint64_t>(__ptr.__property())};
  return _CUDA_VDEV::__load_array<_Np>(
    __ptr.get(), __align, __memory_access, __eviction_policy, __prefetch, __cache_hint);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_HAS_CUDA_COMPILER
#endif // _CUDA___DATA_MOVEMENT_LOAD_H
