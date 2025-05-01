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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER()

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
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__utility/integer_sequence.h>
#  include <cuda/std/array>
#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

/***********************************************************************************************************************
 * CUDA TO PTX MAPPINGS
 **********************************************************************************************************************/

#  define _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, EVICT_POLICY, _PREFETCH, _CACHE_HINT, ...)                           \
    if constexpr ((_PREFETCH) == L2_prefetch_none)                                                                 \
    {                                                                                                              \
      return _CUDA_VPTX::ld##_LOAD_BEHAVIOR##EVICT_POLICY##_CACHE_HINT(_CUDA_VPTX::space_global_t{}, __VA_ARGS__); \
    }                                                                                                              \
    else if constexpr ((_PREFETCH) == L2_prefetch_64B)                                                             \
    {                                                                                                              \
      return _CUDA_VPTX::ld##_LOAD_BEHAVIOR##EVICT_POLICY##_CACHE_HINT##_L2_64B(                                   \
        _CUDA_VPTX::space_global_t{}, __VA_ARGS__);                                                                \
    }                                                                                                              \
    else if constexpr ((_PREFETCH) == L2_prefetch_128B)                                                            \
    {                                                                                                              \
      return _CUDA_VPTX::ld##_LOAD_BEHAVIOR##EVICT_POLICY##_CACHE_HINT##_L2_128B(                                  \
        _CUDA_VPTX::space_global_t{}, __VA_ARGS__);                                                                \
    }                                                                                                              \
    else if constexpr ((_PREFETCH) == L2_prefetch_256B)                                                            \
    {                                                                                                              \
      return _CUDA_VPTX::ld##_LOAD_BEHAVIOR##EVICT_POLICY##_CACHE_HINT##_L2_256B(                                  \
        _CUDA_VPTX::space_global_t{}, __VA_ARGS__);                                                                \
    }

#  define _CCCL_LOAD_ADD_EVICTION_POLICY(_LOAD_BEHAVIOR, _EVICTION_POLICY, _PREFETCH, _CACHE_HINT, ...) \
    if constexpr ((_EVICTION_POLICY) == L1_unchanged_reuse)                                             \
    {                                                                                                   \
      _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, , _PREFETCH, _CACHE_HINT, __VA_ARGS__);                       \
    }                                                                                                   \
    else if constexpr ((_EVICTION_POLICY) == L1_normal_reuse)                                           \
    {                                                                                                   \
      _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, _L1_evict_normal, _PREFETCH, _CACHE_HINT, __VA_ARGS__);       \
    }                                                                                                   \
    else if constexpr ((_EVICTION_POLICY) == L1_low_reuse)                                              \
    {                                                                                                   \
      _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, _L1_evict_first, _PREFETCH, _CACHE_HINT, __VA_ARGS__);        \
    }                                                                                                   \
    else if constexpr ((_EVICTION_POLICY) == L1_high_reuse)                                             \
    {                                                                                                   \
      _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, _L1_evict_last, _PREFETCH, _CACHE_HINT, __VA_ARGS__);         \
    }                                                                                                   \
    else if constexpr ((_EVICTION_POLICY) == L1_unchanged_reuse)                                        \
    {                                                                                                   \
      _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, _L1_evict_unchanged, _PREFETCH, _CACHE_HINT, __VA_ARGS__);    \
    }                                                                                                   \
    else if constexpr ((_EVICTION_POLICY) == L1_no_reuse)                                               \
    {                                                                                                   \
      _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, _L1_no_allocate, _PREFETCH, _CACHE_HINT, __VA_ARGS__);        \
    }

/***********************************************************************************************************************
 * SM-Specific Functions
 **********************************************************************************************************************/

template <typename _Tp, _MemoryAccess _Bp, _L1_ReuseEnum _Ep>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp
__load_sm70(const _Tp* __ptr, __memory_access_t<_Bp> __memory_access, __l1_reuse_t<_Ep> __l1_reuse) noexcept
{
  if constexpr (__memory_access == read_write)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __l1_reuse, L2_prefetch_none, , __ptr);
  }
  else
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __l1_reuse, L2_prefetch_none, , __ptr);
  }
}

template <typename _Tp, _MemoryAccess _Bp, _L1_ReuseEnum _Ep, _L2_PrefetchEnum _Pp>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_sm75(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __l1_reuse_t<_Ep> __l1_reuse,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  if constexpr (__memory_access == read_write && __l2_prefetch == L2_prefetch_256B)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __l1_reuse, L2_prefetch_128B, , __ptr); // fallback
  }
  else if constexpr (__memory_access == read_write)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __l1_reuse, __l2_prefetch, , __ptr);
  }
  else if constexpr (__l2_prefetch == L2_prefetch_256B) // __memory_access == read_only
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __l1_reuse, L2_prefetch_128B, , __ptr); // fallback
  }
  else
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __l1_reuse, __l2_prefetch, , __ptr);
  }
}

template <typename _Tp, _MemoryAccess _Bp, _L1_ReuseEnum _Ep, _L2_PrefetchEnum _Pp, typename _AccessProperty>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_sm80(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __l1_reuse_t<_Ep> __l1_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  if constexpr (__memory_access == read_write && __l2_hint)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __l1_reuse, __l2_prefetch, _L2_cache_hint, __ptr, __l2_hint.__property);
  }
  else if constexpr (__memory_access == read_write && !__l2_hint)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __l1_reuse, __l2_prefetch, , __ptr);
  }
  else if constexpr (__l2_hint) // __memory_access == read_only
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __l1_reuse, __l2_prefetch, _L2_cache_hint, __ptr, __l2_hint.__property);
  }
  else
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __l1_reuse, __l2_prefetch, , __ptr);
  }
}

#  undef _CCCL_LOAD_PTX_CALL
#  undef _CCCL_LOAD_ADD_EVICTION_POLICY

/***********************************************************************************************************************
 * INTERNAL DISPATCH
 **********************************************************************************************************************/

template <typename _Tp, _MemoryAccess _Bp, _L1_ReuseEnum _Ep, _L2_PrefetchEnum _Pp, typename _AccessProperty>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_arch_dispatch(
  const _Tp* __ptr,
  [[maybe_unused]] __memory_access_t<_Bp> __memory_access,
  [[maybe_unused]] __l1_reuse_t<_Ep> __l1_reuse,
  [[maybe_unused]] __l2_hint_t<_AccessProperty> __l2_hint,
  [[maybe_unused]] __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  static_assert(sizeof(_Tp) <= __max_ptx_access_size);
  // clang-format off
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (return _CUDA_VDEV::__load_sm70(__ptr, __memory_access, __l1_reuse);),
    NV_PROVIDES_SM_75, (return _CUDA_VDEV::__load_sm75(__ptr, __memory_access, __l1_reuse, __l2_prefetch);),
    NV_PROVIDES_SM_80, (return _CUDA_VDEV::__load_sm80(__ptr, __memory_access, __l1_reuse, __l2_hint, __l2_prefetch);),
    NV_IS_DEVICE,      (return *__ptr;)); // fallback
  // clang-format on
  _CCCL_UNREACHABLE();
}

/***********************************************************************************************************************
 * UTILITIES
 **********************************************************************************************************************/

template <typename _Tp, _MemoryAccess _Bp, _L1_ReuseEnum _Ep, _L2_PrefetchEnum _Pp, typename _AccessProperty, size_t... _Ip>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, sizeof...(_Ip)> __unroll_load(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __l1_reuse_t<_Ep> __l1_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch,
  _CUDA_VSTD::index_sequence<_Ip...> = {})
{
  _CUDA_VSTD::array<_Tp, sizeof...(_Ip)> __tmp;
  if constexpr (__memory_access == read_write && __l1_reuse == L1_unchanged_reuse && __l2_prefetch == L2_prefetch_none
                && !__l2_hint)
  {
    ((__tmp[_Ip] = __ptr[_Ip]), ...);
  }
  else
  {
    ((__tmp[_Ip] = _CUDA_VDEV::__load_arch_dispatch(__ptr + _Ip, __memory_access, __l1_reuse, __l2_hint, __l2_prefetch)),
     ...);
  }
  return __tmp;
};

/***********************************************************************************************************************
 * INTERNAL API
 **********************************************************************************************************************/

template <typename _Tp, _MemoryAccess _Bp, _L1_ReuseEnum _Ep, _L2_PrefetchEnum _Pp, typename _AccessProperty>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_element(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __l1_reuse_t<_Ep> __l1_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  _CCCL_ASSERT(__ptr != nullptr, "'ptr' must not be null");
  _CCCL_ASSERT(__isGlobal(__ptr), "'ptr' must point to global memory");
  _CCCL_ASSERT(_CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % alignof(_Tp) == 0, "'ptr' must be aligned");
  auto __ptr_gmem = _CUDA_VSTD::bit_cast<const _Tp*>(__cvta_generic_to_global(__ptr));
  if constexpr (__memory_access == read_write && __l1_reuse == L1_unchanged_reuse && __l2_prefetch == L2_prefetch_none
                && !__l2_hint)
  {
    return *__ptr_gmem;
  }
  else
  {
    static_assert(_CUDA_VSTD::has_single_bit(sizeof(_Tp)) || sizeof(_Tp) % __max_ptx_access_size == 0,
                  "'sizeof(_Tp)' must be a power of 2 or multiple of max_alignment with non-default properties");
    constexpr auto __access_size = _CUDA_VSTD::min(sizeof(_Tp), __max_ptx_access_size);
    constexpr auto __num_unroll  = sizeof(_Tp) / __access_size;
    using __aligned_data         = _AlignedData<__access_size>;
    auto __ptr_gmem2             = reinterpret_cast<const __aligned_data*>(__ptr_gmem);
    auto __index_seq             = _CUDA_VSTD::make_index_sequence<__num_unroll>{};
    auto __tmp =
      _CUDA_VDEV::__unroll_load(__ptr_gmem2, __memory_access, __l1_reuse, __l2_hint, __l2_prefetch, __index_seq);
    return _CUDA_VSTD::bit_cast<_Tp>(__tmp);
  }
}

template <size_t _Np,
          typename _Tp,
          size_t _Align,
          _MemoryAccess _Bp,
          _L1_ReuseEnum _Ep,
          _L2_PrefetchEnum _Pp,
          typename _AccessProperty>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, _Np> __load_array(
  const _Tp* __ptr,
  aligned_size_t<_Align>,
  __memory_access_t<_Bp> __memory_access,
  __l1_reuse_t<_Ep> __l1_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  constexpr auto __access_size = sizeof(_Tp) * _Np;
  static_assert(_CUDA_VSTD::has_single_bit(_Align), "_Align must be a power of 2");
  static_assert(_Np > 0);
  static_assert(_Align >= alignof(_Tp), "_Align must be greater than or equal to alignof(_Tp)");
  static_assert(__access_size % _Align == 0, "Np * sizeof(_Tp) must be a multiple of _Align");
  _CCCL_ASSERT(__ptr != nullptr, "'ptr' must not be null");
  _CCCL_ASSERT(_CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % _Align == 0, "'ptr' must be aligned");
  _CCCL_ASSERT(__isGlobal(__ptr), "'ptr' must point to global memory");
  constexpr bool __is_default_access = __memory_access == read_write && __l1_reuse == L1_unchanged_reuse
                                    && __l2_prefetch == L2_prefetch_none && !__l2_hint;
  constexpr auto __max_align  = __is_default_access ? _Align : _CUDA_VSTD::min(_Align, __max_ptx_access_size);
  constexpr auto __num_unroll = __access_size / __max_align;
  using __aligned_data        = _AlignedData<__max_align>;
  auto __ptr_gmem             = _CUDA_VSTD::bit_cast<const __aligned_data*>(__cvta_generic_to_global(__ptr));
  auto __index_seq            = _CUDA_VSTD::make_index_sequence<__num_unroll>{};
  auto __tmp =
    _CUDA_VDEV::__unroll_load(__ptr_gmem, __memory_access, __l1_reuse, __l2_hint, __l2_prefetch, __index_seq);
  using __result_t = _CUDA_VSTD::array<_Tp, _Np>;
  return _CUDA_VSTD::bit_cast<__result_t>(__tmp);
}

/***********************************************************************************************************************
 * USER API
 **********************************************************************************************************************/

template <typename _Tp,
          _MemoryAccess _Bp        = _MemoryAccess::_ReadWrite,
          _L1_ReuseEnum _Ep        = _L1_ReuseEnum::_Unchanged,
          _L2_PrefetchEnum _Pp     = _L2_PrefetchEnum::_None,
          typename _AccessProperty = access_property::global>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp
load(const _Tp* __ptr,
     __memory_access_t<_Bp> __memory_access = read_write,
     __l1_reuse_t<_Ep> __l1_reuse           = L1_unchanged_reuse,
     __l2_prefetch_t<_Pp> __l2_prefetch     = L2_prefetch_none,
     _AccessProperty __access_property      = access_property::global{}) noexcept
{
  return _CUDA_VDEV::__load_element(__ptr, __memory_access, __l1_reuse, __l2_hint_t{__access_property}, __l2_prefetch);
}

template <typename _Tp,
          typename _Prop,
          _MemoryAccess _Bp    = _MemoryAccess::_ReadWrite,
          _L1_ReuseEnum _Ep    = _L1_ReuseEnum::_Unchanged,
          _L2_PrefetchEnum _Pp = _L2_PrefetchEnum::_None>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp
load(annotated_ptr<_Tp, _Prop> __ptr,
     __memory_access_t<_Bp> __memory_access = read_write,
     __l1_reuse_t<_Ep> __l1_reuse           = L1_unchanged_reuse,
     __l2_prefetch_t<_Pp> __l2_prefetch     = L2_prefetch_none) noexcept
{
  return _CUDA_VDEV::__load_element(
    __ptr.__get_raw_ptr(), __memory_access, __l1_reuse, __ptr.__property(), __l2_prefetch);
}

template <size_t _Np,
          typename _Tp,
          size_t _Align            = alignof(_Tp),
          _MemoryAccess _Bp        = _MemoryAccess::_ReadWrite,
          _L1_ReuseEnum _Ep        = _L1_ReuseEnum::_Unchanged,
          _L2_PrefetchEnum _Pp     = _L2_PrefetchEnum::_None,
          typename _AccessProperty = ::cuda::access_property::global>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, _Np>
load(const _Tp* __ptr,
     aligned_size_t<_Align> __align         = aligned_size_t<_Align>{alignof(_Tp)},
     __memory_access_t<_Bp> __memory_access = read_write,
     __l1_reuse_t<_Ep> __l1_reuse           = L1_unchanged_reuse,
     __l2_prefetch_t<_Pp> __l2_prefetch     = L2_prefetch_none,
     _AccessProperty __access_property      = ::cuda::access_property::global{}) noexcept
{
  return _CUDA_VDEV::__load_array<_Np>(
    __ptr, __align, __memory_access, __l1_reuse, __l2_hint_t{__access_property}, __l2_prefetch);
}

template <size_t _Np,
          typename _Tp,
          typename _Prop,
          size_t _Align        = alignof(_Tp),
          _MemoryAccess _Bp    = _MemoryAccess::_ReadWrite,
          _L1_ReuseEnum _Ep    = _L1_ReuseEnum::_Unchanged,
          _L2_PrefetchEnum _Pp = _L2_PrefetchEnum::_None>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, _Np>
load(annotated_ptr<_Tp, _Prop> __ptr,
     aligned_size_t<_Align> __align         = aligned_size_t<_Align>{alignof(_Tp)},
     __memory_access_t<_Bp> __memory_access = read_write,
     __l1_reuse_t<_Ep> __l1_reuse           = L1_unchanged_reuse,
     __l2_prefetch_t<_Pp> __l2_prefetch     = L2_prefetch_none) noexcept
{
  return _CUDA_VDEV::__load_array<_Np>(
    __ptr.__get_raw_ptr(), __align, __memory_access, __l1_reuse, __ptr.__property(), __l2_prefetch);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_HAS_CUDA_COMPILER()
#endif // _CUDA___DATA_MOVEMENT_LOAD_H
