//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DATA_MOVEMENT_STORE_H
#define _CUDA___DATA_MOVEMENT_STORE_H

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
#  include <cuda/__cmath/pow2.h>
#  include <cuda/__data_movement/aligned_data.h>
#  include <cuda/__data_movement/properties.h>
#  include <cuda/__memory/is_aligned.h>
#  include <cuda/__ptx/instructions/st.h>
#  include <cuda/__utility/static_for.h>
#  include <cuda/annotated_ptr>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__bit/bit_cast.h>
#  include <cuda/std/__cccl/unreachable.h>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/array>
#  include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

/***********************************************************************************************************************
 * CUDA TO PTX MAPPINGS
 **********************************************************************************************************************/

#  define _CCCL_STORE_PTX_CALL(_L1_POLICY, _L2_POLICY, _L2_HINT, ...)                                    \
    if constexpr (_L2_POLICY == cache_reuse_unchanged)                                                   \
    {                                                                                                    \
      _CUDA_VPTX::st##_L1_POLICY##_L2_HINT(_CUDA_VPTX::space_global_t{}, __VA_ARGS__);                   \
    }                                                                                                    \
    else if constexpr (_L2_POLICY == cache_reuse_normal)                                                 \
    {                                                                                                    \
      _CUDA_VPTX::st##_L1_POLICY##_L2_evict_normal##_L2_HINT(_CUDA_VPTX::space_global_t{}, __VA_ARGS__); \
    }                                                                                                    \
    else if constexpr (_L2_POLICY == cache_reuse_low)                                                    \
    {                                                                                                    \
      _CUDA_VPTX::st##_L1_POLICY##_L2_evict_first##_L2_HINT(_CUDA_VPTX::space_global_t{}, __VA_ARGS__);  \
    }                                                                                                    \
    else if constexpr (_L2_POLICY == cache_reuse_high)                                                   \
    {                                                                                                    \
      _CUDA_VPTX::st##_L1_POLICY##_L2_evict_last##_L2_HINT(_CUDA_VPTX::space_global_t{}, __VA_ARGS__);   \
    }

#  define _CCCL_STORE_ADD_L1_POLICY(_L1_POLICY, _L2_POLICY, _L2_HINT, ...)       \
    if constexpr (_L1_POLICY == cache_reuse_unchanged)                           \
    {                                                                            \
      _CCCL_STORE_PTX_CALL(, _L2_POLICY, _L2_HINT, __VA_ARGS__);                 \
    }                                                                            \
    else if constexpr (_L1_POLICY == cache_reuse_normal)                         \
    {                                                                            \
      _CCCL_STORE_PTX_CALL(_L1_evict_normal, _L2_POLICY, _L2_HINT, __VA_ARGS__); \
    }                                                                            \
    else if constexpr (_L1_POLICY == cache_reuse_low)                            \
    {                                                                            \
      _CCCL_STORE_PTX_CALL(_L1_evict_first, _L2_POLICY, _L2_HINT, __VA_ARGS__);  \
    }                                                                            \
    else if constexpr (_L1_POLICY == cache_reuse_high)                           \
    {                                                                            \
      _CCCL_STORE_PTX_CALL(_L1_evict_last, _L2_POLICY, _L2_HINT, __VA_ARGS__);   \
    }                                                                            \
    else if constexpr (_L1_POLICY == cache_no_reuse)                             \
    {                                                                            \
      _CCCL_STORE_PTX_CALL(_L1_no_allocate, _L2_POLICY, _L2_HINT, __VA_ARGS__);  \
    }

/***********************************************************************************************************************
 * SM-Specific Functions
 **********************************************************************************************************************/

template <typename _Tp, _CacheReuseEnum _L1>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __store_sm70(_Tp* __ptr, _Tp __data, __cache_reuse_t<_L1> __l1_reuse) noexcept
{
  if constexpr (__l1_reuse == cache_reuse_unchanged)
  {
    *__ptr = __data;
  }
  else
  {
    _CCCL_STORE_ADD_L1_POLICY(__l1_reuse, /*l2_reuse*/, /*l2_hint*/, __ptr, __data);
  }
}

template <typename _Tp, _CacheReuseEnum _L1, typename _AccessProperty>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __store_sm80(
  _Tp* __ptr,
  _Tp __data,
  __cache_reuse_t<_L1> __l1_reuse,
  [[maybe_unused]] __l2_hint_t<_AccessProperty> __l2_hint) noexcept
{
  if constexpr (!__l2_hint)
  {
    _CUDA_DEVICE::__store_sm70(__data, __ptr, __l1_reuse);
  }
  else
  {
    _CCCL_STORE_ADD_L1_POLICY(__l1_reuse, /*l2_reuse*/, _L2_cache_hint, __ptr, __data, __l2_hint.__property);
  }
}

template <typename _Tp, _CacheReuseEnum _L1, _CacheReuseEnum _L2, typename _AccessProperty>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __store_sm100(
  _Tp* __ptr,
  _Tp __data,
  __cache_reuse_t<_L1> __l1_reuse,
  __cache_reuse_t<_L2> __l2_reuse,
  [[maybe_unused]] __l2_hint_t<_AccessProperty> __l2_hint) noexcept
{
  if constexpr (__l2_reuse == cache_reuse_unchanged || sizeof(_Tp) <= 8)
  {
    _CUDA_DEVICE::__store_sm80(__data, __ptr, __l1_reuse, __l2_hint);
  }
  else
  {
    _CCCL_STORE_ADD_L1_POLICY(__l1_reuse, __l2_reuse, _L2_cache_hint, __ptr, __data, __l2_hint.__property);
  }
}

/***********************************************************************************************************************
 * INTERNAL DISPATCH
 **********************************************************************************************************************/

template <typename _Tp, _CacheReuseEnum _L1, _CacheReuseEnum _L2, typename _AccessProperty>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __store_dispatch(
  _Tp __data,
  _Tp* __ptr,
  [[maybe_unused]] __cache_reuse_t<_L1> __l1_reuse,
  [[maybe_unused]] __cache_reuse_t<_L2> __l2_reuse,
  [[maybe_unused]] __l2_hint_t<_AccessProperty> __l2_hint) noexcept
{
  // clang-format off
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_100, (_CUDA_DEVICE::__store_sm100(__ptr, __data, __l1_reuse, __l2_reuse, __l2_hint);),
                     NV_PROVIDES_SM_80,  (_CUDA_DEVICE::__store_sm80(__ptr,  __data, __l1_reuse, __l2_hint);),
                     NV_PROVIDES_SM_70,  (_CUDA_DEVICE::__store_sm70(__ptr,  __data, __l1_reuse);),
                     NV_IS_DEVICE,       (*__ptr = __data;)); // fallback
  // clang-format on
  _CCCL_UNREACHABLE();
}

/***********************************************************************************************************************
 * UTILITIES
 **********************************************************************************************************************/

// template <typename _Tp, size_t _Np, _CacheReuseEnum _L1, _CacheReuseEnum _L2, typename _AccessProperty, size_t...
// _Ip> _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __unroll_store(
//   const _CUDA_VSTD::array<_Tp, _Np>& data,
//   _Tp* __ptr,
//   __cache_reuse_t<_L1> __l1_reuse,
//   __cache_reuse_t<_L2> __l2_reuse,
//   __l2_hint_t<_AccessProperty> __l2_hint,
//   _CUDA_VSTD::index_sequence<_Ip...> = {})
//{
//   ((, ...);
// }

/***********************************************************************************************************************
 * INTERNAL API
 **********************************************************************************************************************/

template <size_t _MaxPtxAccessSize,
          typename _Tp,
          size_t _Align,
          _CacheReuseEnum _L1,
          _CacheReuseEnum _L2,
          typename _AccessProperty>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __store_impl(
  _Tp* __ptr,
  aligned_size_t<_Align>,
  _Tp __data,
  __cache_reuse_t<_L1> __l1_reuse,
  __cache_reuse_t<_L2> __l2_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint) noexcept
{
  static_assert(!_CUDA_VSTD::is_const_v<_Tp>, "_Tp must not be const");
  _CCCL_ASSERT(__ptr != nullptr, "'ptr' must not be null");
  _CCCL_ASSERT(__isGlobal(__ptr), "'ptr' must point to global memory");
  _CCCL_ASSERT(::cuda::is_aligned(__ptr, _Align));
  constexpr auto __max_align = _CUDA_VSTD::min({_Align, _MaxPtxAccessSize, sizeof(_Tp)});
  static_assert(sizeof(_Tp) % __max_align == 0);
  static_assert(::cuda::is_power_of_two(__max_align), "sizeof(_Tp) must be a power of 2 for overaligned types");
  constexpr auto __num_unroll = sizeof(_Tp) / __max_align;
  using __aligned_data        = _AlignedData<__max_align>;
  auto __ptr_gmem             = _CUDA_VSTD::bit_cast<__aligned_data*>(__cvta_generic_to_global(__ptr));
  using __store_type          = _CUDA_VSTD::array<__aligned_data, __num_unroll>;
  auto __index_seq            = _CUDA_VSTD::make_index_sequence<__num_unroll>{};
  auto __data_tmp             = _CUDA_VSTD::bit_cast<__store_type>(__data);
  ::cuda::static_for<__num_unroll>([&](auto index) {
    _CUDA_DEVICE::__store_dispatch(__ptr + index, __data_tmp[index], __l1_reuse, __l2_reuse, __l2_hint);
  });
  //_CUDA_DEVICE::__unroll_store(__ptr_gmem, __data_tmp, __l1_reuse, __l2_reuse, __l2_hint, __index_seq);
}

template <typename _Tp,
          size_t _Align,
          _MemoryAccess _Bp,
          _CacheReuseEnum _L1,
          _CacheReuseEnum _L2,
          _L2_PrefetchEnum _Pp,
          typename _AccessProperty>
[[nodiscard]] _CCCL_PURE _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __store_ptx_isa_dispatch(
  const _Tp* __ptr,
  aligned_size_t<_Align> __align,
  __memory_access_t<_Bp> __memory_access,
  __cache_reuse_t<_L1> __l1_reuse,
  [[maybe_unused]] __cache_reuse_t<_L2> __l2_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  __l2_prefetch_t<_Pp> __l2_prefetch) noexcept
{
  if constexpr (__cccl_ptx_isa >= 880)
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_100,
      (return _CUDA_DEVICE::__store_impl<32>(
                __ptr, __align, __memory_access, __l1_reuse, __l2_reuse, __l2_hint, __l2_prefetch);),
      (return _CUDA_DEVICE::__store_impl<__max_ptx_access_size>(
                __ptr, __align, __memory_access, __l1_reuse, __l2_reuse, __l2_hint, __l2_prefetch);))
  }
  else
  {
    return _CUDA_DEVICE::__store_impl<__max_ptx_access_size>(
      __ptr, __align, __memory_access, __l1_reuse, cache_reuse_unchanged, __l2_hint, __l2_prefetch);
  }
}

template <size_t _Np, typename _Tp, size_t _Align, _CacheReuseEnum _L1, _CacheReuseEnum _L2, typename _AccessProperty>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __store_array(
  _Tp* __ptr,
  _CUDA_VSTD::array<_Tp, _Np> __data,
  aligned_size_t<_Align> __align,
  __cache_reuse_t<_L1> __l1_reuse,
  __cache_reuse_t<_L2> __l2_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint) noexcept
{
  static_assert(_Np > 0);
  static_assert(_Align >= alignof(_Tp), "_Align must be greater than or equal to alignof(_Tp)");
  using __result_t = _CUDA_VSTD::array<_Tp, _Np>;
  auto __ptr1      = reinterpret_cast<const __result_t*>(__ptr);
  return _CUDA_DEVICE::__store_ptx_isa_dispatch(__ptr1, __data, __align, __l1_reuse, __l2_reuse, __l2_hint);
}

/***********************************************************************************************************************
 * USER API
 **********************************************************************************************************************/

template <typename _Tp,
          _CacheReuseEnum _L1      = _CacheReuseEnum::_Unchanged,
          _CacheReuseEnum _L2      = _CacheReuseEnum::_Unchanged,
          typename _AccessProperty = access_property::global>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(_Tp* __ptr,
      _Tp __data,
      __cache_reuse_t<_L1> __l1_reuse = cache_reuse_unchanged,
      __cache_reuse_t<_L2> __l2_reuse = cache_reuse_unchanged,
      _AccessProperty __l2_hint       = access_property::global{}) noexcept
{
  _CUDA_DEVICE::__store_ptx_isa_dispatch(__ptr, __data, __l1_reuse, __l2_reuse, __l2_hint_t{__l2_hint});
}

template <typename _Tp,
          typename _Prop,
          _CacheReuseEnum _L1 = _CacheReuseEnum::_Unchanged,
          _CacheReuseEnum _L2 = _CacheReuseEnum::_Unchanged>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(annotated_ptr<_Tp, _Prop> __ptr,
      _Tp __data,
      __cache_reuse_t<_L1> __l1_reuse = cache_reuse_unchanged,
      __cache_reuse_t<_L1> __l2_reuse = cache_reuse_unchanged) noexcept
{
  _CUDA_DEVICE::__store_ptx_isa_dispatch(__ptr.__get_raw_ptr(), __data, __l1_reuse, __l2_reuse, __ptr.__property());
}

template <size_t _Np,
          typename _Tp,
          size_t _Align            = alignof(_Tp),
          _CacheReuseEnum _L1      = _CacheReuseEnum::_Unchanged,
          _CacheReuseEnum _L2      = _CacheReuseEnum::_Unchanged,
          typename _AccessProperty = access_property::global>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(_Tp* __ptr,
      const _CUDA_VSTD::array<_Tp, _Np>& __data,
      aligned_size_t<_Align> __align  = aligned_size_t<_Align>{alignof(_Tp)},
      __cache_reuse_t<_L1> __l1_reuse = cache_reuse_unchanged,
      __cache_reuse_t<_L1> __l2_reuse = cache_reuse_unchanged,
      _AccessProperty __l2_hint       = access_property::global{}) noexcept
{
  _CUDA_DEVICE::__store_array<_Np>(__ptr, __data, __align, __l1_reuse, __l2_reuse, __l2_hint_t{__l2_hint});
}

template <size_t _Np,
          typename _Tp,
          typename _Prop,
          size_t _Align            = alignof(_Tp),
          _CacheReuseEnum _L1      = _CacheReuseEnum::_Unchanged,
          _CacheReuseEnum _L2      = _CacheReuseEnum::_Unchanged,
          typename _AccessProperty = access_property::global>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(annotated_ptr<_Tp, _Prop> __ptr,
      const _CUDA_VSTD::array<_Tp, _Np>& __data,
      aligned_size_t<_Align> __align  = aligned_size_t<_Align>{alignof(_Tp)},
      __cache_reuse_t<_L1> __l1_reuse = cache_reuse_unchanged,
      __cache_reuse_t<_L2> __l2_reuse = cache_reuse_unchanged) noexcept
{
  _CUDA_DEVICE::__store_array<_Np>(__ptr.__get_raw_ptr(), __data, __align, __l1_reuse, __l2_reuse, __ptr.__property());
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_HAS_CUDA_COMPILER
#endif // _CUDA___DATA_MOVEMENT_STORE_H
