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
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__utility/integer_sequence.h>
#  include <cuda/std/array>
#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

/***********************************************************************************************************************
 * CUDA TO PTX MAPPINGS
 **********************************************************************************************************************/

#  define _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, _L1_POLICY, _L2_POLICY, _CACHE_HINT, _PREFETCH, ...) \
    if constexpr ((_PREFETCH) == L2_prefetch_none)                                                 \
    {                                                                                              \
      return _CUDA_VPTX::ld##_LOAD_BEHAVIOR##_L1_POLICY##_L2_POLICY##_CACHE_HINT(                  \
        _CUDA_VPTX::space_global_t{}, __VA_ARGS__);                                                \
    }                                                                                              \
    else if constexpr ((_PREFETCH) == L2_prefetch_64B)                                             \
    {                                                                                              \
      return _CUDA_VPTX::ld##_LOAD_BEHAVIOR##_L1_POLICY##_L2_POLICY##_CACHE_HINT##_L2_64B(         \
        _CUDA_VPTX::space_global_t{}, __VA_ARGS__);                                                \
    }                                                                                              \
    else if constexpr ((_PREFETCH) == L2_prefetch_128B)                                            \
    {                                                                                              \
      return _CUDA_VPTX::ld##_LOAD_BEHAVIOR##_L1_POLICY##_L2_POLICY##_CACHE_HINT##_L2_128B(        \
        _CUDA_VPTX::space_global_t{}, __VA_ARGS__);                                                \
    }                                                                                              \
    else if constexpr ((_PREFETCH) == L2_prefetch_256B)                                            \
    {                                                                                              \
      return _CUDA_VPTX::ld##_LOAD_BEHAVIOR##_L1_POLICY##_L2_POLICY##_CACHE_HINT##_L2_256B(        \
        _CUDA_VPTX::space_global_t{}, __VA_ARGS__);                                                \
    }

#  define _CCCL_LOAD_ADD_L2_POLICY(_LOAD_BEHAVIOR, _L1_POLICY, _L2_POLICY, _CACHE_HINT, _PREFETCH, ...)       \
    if constexpr ((_L2_POLICY) == cache_reuse_unchanged)                                                      \
    {                                                                                                         \
      _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, _L1_POLICY, , _CACHE_HINT, _PREFETCH, __VA_ARGS__);                 \
    }                                                                                                         \
    else if constexpr ((_L2_POLICY) == cache_reuse_normal)                                                    \
    {                                                                                                         \
      _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, _L1_POLICY, _L2_evict_normal, _CACHE_HINT, _PREFETCH, __VA_ARGS__); \
    }                                                                                                         \
    else if constexpr ((_L2_POLICY) == cache_reuse_low)                                                       \
    {                                                                                                         \
      _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, _L1_POLICY, _L2_evict_first, _CACHE_HINT, _PREFETCH, __VA_ARGS__);  \
    }                                                                                                         \
    else if constexpr ((_L2_POLICY) == cache_reuse_high)                                                      \
    {                                                                                                         \
      _CCCL_LOAD_PTX_CALL(_LOAD_BEHAVIOR, _L1_POLICY, _L2_evict_last, _CACHE_HINT, _PREFETCH, __VA_ARGS__);   \
    }

#  define _CCCL_LOAD_ADD_L1_POLICY(_LOAD_BEHAVIOR, _L1_POLICY, _L2_POLICY, _CACHE_HINT, _PREFETCH, ...)            \
    if constexpr ((_L1_POLICY) == cache_reuse_unchanged)                                                           \
    {                                                                                                              \
      _CCCL_LOAD_ADD_L2_POLICY(_LOAD_BEHAVIOR, , _L2_POLICY, _CACHE_HINT, _PREFETCH, __VA_ARGS__);                 \
    }                                                                                                              \
    else if constexpr ((_L1_POLICY) == cache_reuse_normal)                                                         \
    {                                                                                                              \
      _CCCL_LOAD_ADD_L2_POLICY(_LOAD_BEHAVIOR, _L1_evict_normal, _L2_POLICY, _CACHE_HINT, _PREFETCH, __VA_ARGS__); \
    }                                                                                                              \
    else if constexpr ((_L1_POLICY) == cache_reuse_low)                                                            \
    {                                                                                                              \
      _CCCL_LOAD_ADD_L2_POLICY(_LOAD_BEHAVIOR, _L1_evict_first, _L2_POLICY, _CACHE_HINT, _PREFETCH, __VA_ARGS__);  \
    }                                                                                                              \
    else if constexpr ((_L1_POLICY) == cache_reuse_high)                                                           \
    {                                                                                                              \
      _CCCL_LOAD_ADD_L2_POLICY(_LOAD_BEHAVIOR, _L1_evict_last, _L2_POLICY, _CACHE_HINT, _PREFETCH, __VA_ARGS__);   \
    }                                                                                                              \
    else if constexpr ((_L1_POLICY) == cache_no_reuse)                                                             \
    {                                                                                                              \
      _CCCL_LOAD_ADD_L2_POLICY(_LOAD_BEHAVIOR, _L1_no_allocate, _L2_POLICY, _CACHE_HINT, _PREFETCH, __VA_ARGS__);  \
    }

/***********************************************************************************************************************
 * SM-Specific Functions
 **********************************************************************************************************************/

template <typename _Tp, _MemoryAccess _Bp, _CacheReuseEnum _L1>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp
__load_sm70(const _Tp* __ptr, __memory_access_t<_Bp> __memory_access, __cache_reuse_t<_L1> __l1_reuse) noexcept
{
  if constexpr (__memory_access == read_write && __l1_reuse == cache_no_reuse)
  {
    return *__ptr; // no ptx
  }
  else if constexpr (__memory_access == read_write)
  {
    _CCCL_LOAD_ADD_L1_POLICY(, __l1_reuse, /*l2_reuse*/, /*l2_hint*/, L2_prefetch_none, __ptr);
  }
  else
  {
    _CCCL_LOAD_ADD_L1_POLICY(_nc, __l1_reuse, /*l2_reuse*/, /*l2_hint*/, L2_prefetch_none, __ptr);
  }
}

template <typename _Tp, _MemoryAccess _Bp, _CacheReuseEnum _L1, _L2_PrefetchEnum _Pp>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_sm75(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __cache_reuse_t<_L1> __l1_reuse,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  if constexpr (__l2_prefetch == L2_prefetch_none)
  {
    return _CUDA_VDEV::__load_sm70(__ptr, __memory_access, __l1_reuse);
  }
  else if constexpr (__l2_prefetch == L2_prefetch_256B)
  {
    return _CUDA_VDEV::__load_sm75(__ptr, __memory_access, __l1_reuse, L2_prefetch_128B); // fallback
  }
  else if constexpr (__memory_access == read_write)
  {
    _CCCL_LOAD_ADD_L1_POLICY(/*read,write*/, __l1_reuse, __l2_prefetch, /*l2_hint*/, __ptr);
  }
  else
  {
    _CCCL_LOAD_ADD_L1_POLICY(_nc, __l1_reuse, __l2_prefetch, /*l2_hint*/, __ptr);
  }
}

template <typename _Tp, _MemoryAccess _Bp, _CacheReuseEnum _L1, typename _AccessProperty, _L2_PrefetchEnum _Pp>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_sm80(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __cache_reuse_t<_L1> __l1_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  if constexpr (!__l2_hint)
  {
    return _CUDA_VDEV::__load_sm75(__ptr, __memory_access, __l1_reuse, __l2_prefetch);
  }
  else if constexpr (__memory_access == read_write)
  {
    _CCCL_LOAD_ADD_L1_POLICY(/*read,write*/, __l1_reuse, _L2_cache_hint, __l2_prefetch, __ptr, __l2_hint.__property);
  }
  else // __memory_access == read_only
  {
    _CCCL_LOAD_ADD_L1_POLICY(_nc, __l1_reuse, _L2_cache_hint, __l2_prefetch, __ptr, __l2_hint.__property);
  }
}

template <typename _Tp,
          _MemoryAccess _Bp,
          _CacheReuseEnum _L1,
          _CacheReuseEnum _L2,
          typename _AccessProperty,
          _L2_PrefetchEnum _Pp>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_sm100(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __cache_reuse_t<_L1> __l1_reuse,
  __cache_reuse_t<_L2> __l2_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  if constexpr (__l2_reuse == cache_reuse_unchanged || sizeof(_Tp) <= 8)
  {
    return __load_sm80(__ptr, __memory_access, __l1_reuse, __l2_hint, __l2_prefetch);
  }
  else
  {
    if constexpr (__memory_access == read_write)
    {
      _CCCL_LOAD_ADD_L1_POLICY(, __l1_reuse, __l2_reuse, _L2_cache_hint, __l2_prefetch, __ptr, __l2_hint.__property);
    }
    else
    {
      _CCCL_LOAD_ADD_L1_POLICY(_nc, __l1_reuse, __l2_reuse, _L2_cache_hint, __l2_prefetch, __ptr, __l2_hint.__property);
    }
  }
}

#  undef _CCCL_LOAD_PTX_CALL
#  undef _CCCL_LOAD_ADD_L1_POLICY

/***********************************************************************************************************************
 * INTERNAL DISPATCH
 **********************************************************************************************************************/

template <typename _Tp,
          _MemoryAccess _Bp,
          _CacheReuseEnum _L1,
          _CacheReuseEnum _L2,
          _L2_PrefetchEnum _Pp,
          typename _AccessProperty>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_arch_dispatch(
  const _Tp* __ptr,
  [[maybe_unused]] __memory_access_t<_Bp> __memory_access,
  [[maybe_unused]] __cache_reuse_t<_L1> __l1_reuse,
  [[maybe_unused]] __cache_reuse_t<_L2> __l2_reuse,
  [[maybe_unused]] __l2_hint_t<_AccessProperty> __l2_hint,
  [[maybe_unused]] __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  // clang-format off
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_100, (return _CUDA_VDEV::__load_sm100(__ptr, __memory_access, __l1_reuse, __l2_reuse, __l2_hint,
                                                         __l2_prefetch);),
    NV_PROVIDES_SM_80,  (return _CUDA_VDEV::__load_sm80(__ptr, __memory_access, __l1_reuse, __l2_hint, __l2_prefetch);),
    NV_PROVIDES_SM_75,  (return _CUDA_VDEV::__load_sm75(__ptr, __memory_access, __l1_reuse, __l2_prefetch);),
    NV_PROVIDES_SM_70,  (return _CUDA_VDEV::__load_sm70(__ptr, __memory_access, __l1_reuse);),
    NV_IS_DEVICE,       (return *__ptr;)); // fallback
  // clang-format on
  _CCCL_UNREACHABLE();
}

/***********************************************************************************************************************
 * UTILITIES
 **********************************************************************************************************************/

template <typename _Tp,
          _MemoryAccess _Bp,
          _CacheReuseEnum _L1,
          _CacheReuseEnum _L2,
          _L2_PrefetchEnum _Pp,
          typename _AccessProperty,
          size_t... _Ip>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, sizeof...(_Ip)> __unroll_load(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access,
  __cache_reuse_t<_L1> __l1_reuse,
  __cache_reuse_t<_L2> __l2_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch,
  _CUDA_VSTD::index_sequence<_Ip...> = {})
{
  _CUDA_VSTD::array<_Tp, sizeof...(_Ip)> __tmp;
  ((__tmp[_Ip] =
      _CUDA_VDEV::__load_arch_dispatch(__ptr + _Ip, __memory_access, __l1_reuse, __l2_reuse, __l2_hint, __l2_prefetch)),
   ...);
  return __tmp;
};

/***********************************************************************************************************************
 * INTERNAL API
 **********************************************************************************************************************/

template <size_t _MaxPtxAccessSize,
          typename _Tp,
          size_t _Align,
          _MemoryAccess _Bp,
          _CacheReuseEnum _L1,
          _CacheReuseEnum _L2,
          _L2_PrefetchEnum _Pp,
          typename _AccessProperty>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_impl(
  const _Tp* __ptr,
  aligned_size_t<_Align>,
  __memory_access_t<_Bp> __memory_access,
  __cache_reuse_t<_L1> __l1_reuse,
  __cache_reuse_t<_L2> __l2_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  static_assert(_CUDA_VSTD::has_single_bit(_Align), "_Align must be a power of 2");
  _CCCL_ASSERT(__ptr != nullptr, "'ptr' must not be null");
  _CCCL_ASSERT(__isGlobal(__ptr), "'ptr' must point to global memory");
  _CCCL_ASSERT(_CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % _Align == 0, "'ptr' must be aligned");
  constexpr auto __max_align = _CUDA_VSTD::min({_Align, _MaxPtxAccessSize, sizeof(_Tp)});
  static_assert(sizeof(_Tp) % __max_align == 0);
  constexpr auto __num_unroll = sizeof(_Tp) / __max_align;
  using __aligned_data_t      = _AlignedData<__max_align>;
  auto __ptr_gmem             = _CUDA_VSTD::bit_cast<const __aligned_data_t*>(__cvta_generic_to_global(__ptr));
  auto __index_seq            = _CUDA_VSTD::make_index_sequence<__num_unroll>{};
  auto __tmp                  = _CUDA_VDEV::__unroll_load(
    __ptr_gmem, __memory_access, __l1_reuse, __l2_reuse, __l2_hint, __l2_prefetch, __index_seq);
  return _CUDA_VSTD::bit_cast<_Tp>(__tmp);
}

template <typename _Tp,
          size_t _Align,
          _MemoryAccess _Bp,
          _CacheReuseEnum _L1,
          _CacheReuseEnum _L2,
          _L2_PrefetchEnum _Pp,
          typename _AccessProperty>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_ptx_isa_dispatch(
  const _Tp* __ptr,
  aligned_size_t<_Align> __align,
  __memory_access_t<_Bp> __memory_access,
  __cache_reuse_t<_L1> __l1_reuse,
  __cache_reuse_t<_L2> __l2_reuse,
  [[maybe_unused]] __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  if constexpr (__cccl_ptx_isa >= 880)
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_100,
      (return _CUDA_VDEV::__load_impl<32>(
                __ptr, __align, __memory_access, __l1_reuse, __l2_reuse, __l2_hint, __l2_prefetch);),
      (return _CUDA_VDEV::__load_impl<__max_ptx_access_size>(
                __ptr, __align, __memory_access, __l1_reuse, __l2_reuse, __l2_hint, __l2_prefetch);))
  }
  else
  {
    return _CUDA_VDEV::__load_impl<__max_ptx_access_size>(
      __ptr, __align, __memory_access, __l1_reuse, cache_reuse_unchanged, __l2_hint, __l2_prefetch);
  }
}

template <size_t _Np,
          typename _Tp,
          size_t _Align,
          _MemoryAccess _Bp,
          _CacheReuseEnum _L1,
          _CacheReuseEnum _L2,
          _L2_PrefetchEnum _Pp,
          typename _AccessProperty>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, _Np> __load_array(
  const _Tp* __ptr,
  aligned_size_t<_Align> __align,
  __memory_access_t<_Bp> __memory_access,
  __cache_reuse_t<_L1> __l1_reuse,
  __cache_reuse_t<_L2> __l2_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  static_assert(_Np > 0);
  static_assert(_Align >= alignof(_Tp), "_Align must be greater than or equal to alignof(_Tp)");
  using __result_t = _CUDA_VSTD::array<_Tp, _Np>;
  auto __ptr1      = reinterpret_cast<const __result_t*>(__ptr);
  return _CUDA_VDEV::__load_ptx_isa_dispatch(
    __ptr1, __align, __memory_access, __l1_reuse, __l2_reuse, __l2_hint, __l2_prefetch);
}

/***********************************************************************************************************************
 * USER API
 **********************************************************************************************************************/

template <typename _Tp,
          _MemoryAccess _Bp        = _MemoryAccess::_ReadWrite,
          _CacheReuseEnum _L1      = _CacheReuseEnum::_Unchanged,
          _CacheReuseEnum _L2      = _CacheReuseEnum::_Unchanged,
          typename _AccessProperty = access_property::global,
          _L2_PrefetchEnum _Pp     = _L2_PrefetchEnum::_None>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp load(
  const _Tp* __ptr,
  __memory_access_t<_Bp> __memory_access = read_write,
  __cache_reuse_t<_L1> __l1_reuse        = cache_reuse_unchanged,
  __cache_reuse_t<_L2> __l2_reuse        = cache_reuse_unchanged,
  _AccessProperty __l2_hint              = access_property::global{},
  __l2_prefetch_t<_Pp> __l2_prefetch     = L2_prefetch_none) noexcept
{
  constexpr auto __align = aligned_size_t<alignof(_Tp)>{};
  return _CUDA_VDEV::__load_ptx_isa_dispatch(
    __ptr, __align, __memory_access, __l1_reuse, __l2_reuse, __l2_hint_t{__l2_hint}, __l2_prefetch);
}

template <typename _Tp,
          typename _Prop,
          _MemoryAccess _Bp    = _MemoryAccess::_ReadWrite,
          _CacheReuseEnum _L1  = _CacheReuseEnum::_Unchanged,
          _CacheReuseEnum _L2  = _CacheReuseEnum::_Unchanged,
          _L2_PrefetchEnum _Pp = _L2_PrefetchEnum::_None>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp
load(annotated_ptr<_Tp, _Prop> __ptr,
     __memory_access_t<_Bp> __memory_access = read_write,
     __cache_reuse_t<_L1> __l1_reuse        = cache_reuse_unchanged,
     __cache_reuse_t<_L2> __l2_reuse        = cache_reuse_unchanged,
     __l2_prefetch_t<_Pp> __l2_prefetch     = L2_prefetch_none) noexcept
{
  constexpr auto __align = aligned_size_t<alignof(_Tp)>{};
  return _CUDA_VDEV::__load_ptx_isa_dispatch(
    __ptr.__get_raw_ptr(), __align, __memory_access, __l1_reuse, __l2_reuse, __ptr.__property(), __l2_prefetch);
}

template <size_t _Np,
          typename _Tp,
          size_t _Align            = alignof(_Tp),
          _MemoryAccess _Bp        = _MemoryAccess::_ReadWrite,
          _CacheReuseEnum _L1      = _CacheReuseEnum::_Unchanged,
          _CacheReuseEnum _L2      = _CacheReuseEnum::_Unchanged,
          typename _AccessProperty = ::cuda::access_property::global,
          _L2_PrefetchEnum _Pp     = _L2_PrefetchEnum::_None>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, _Np>
load(const _Tp* __ptr,
     aligned_size_t<_Align> __align         = aligned_size_t<_Align>{alignof(_Tp)},
     __memory_access_t<_Bp> __memory_access = read_write,
     __cache_reuse_t<_L1> __l1_reuse        = cache_reuse_unchanged,
     __cache_reuse_t<_L2> __l2_reuse        = cache_reuse_unchanged,
     _AccessProperty __l2_hint              = ::cuda::access_property::global{},
     __l2_prefetch_t<_Pp> __l2_prefetch     = L2_prefetch_none) noexcept
{
  return _CUDA_VDEV::__load_array<_Np>(
    __ptr, __align, __memory_access, __l1_reuse, __l2_reuse, __l2_hint_t{__l2_hint}, __l2_prefetch);
}

template <size_t _Np,
          typename _Tp,
          typename _Prop,
          size_t _Align        = alignof(_Tp),
          _MemoryAccess _Bp    = _MemoryAccess::_ReadWrite,
          _CacheReuseEnum _L1  = _CacheReuseEnum::_Unchanged,
          _CacheReuseEnum _L2  = _CacheReuseEnum::_Unchanged,
          _L2_PrefetchEnum _Pp = _L2_PrefetchEnum::_None>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, _Np>
load(annotated_ptr<_Tp, _Prop> __ptr,
     aligned_size_t<_Align> __align         = aligned_size_t<_Align>{alignof(_Tp)},
     __memory_access_t<_Bp> __memory_access = read_write,
     __cache_reuse_t<_L1> __l1_reuse        = cache_reuse_unchanged,
     __cache_reuse_t<_L2> __l2_reuse        = cache_reuse_unchanged,
     __l2_prefetch_t<_Pp> __l2_prefetch     = L2_prefetch_none) noexcept
{
  return _CUDA_VDEV::__load_array<_Np>(
    __ptr.__get_raw_ptr(), __align, __memory_access, __l1_reuse, __l2_reuse, __ptr.__property(), __l2_prefetch);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_HAS_CUDA_COMPILER()
#endif // _CUDA___DATA_MOVEMENT_LOAD_H
