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
#  include <cuda/__data_movement/aligned_data.h>
#  include <cuda/__data_movement/properties.h>
#  include <cuda/__ptx/instructions/st.h>
#  include <cuda/annotated_ptr>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__bit/has_single_bit.h>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/array>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

/***********************************************************************************************************************
 * CUDA TO PTX MAPPINGS
 **********************************************************************************************************************/

#  define _CCCL_STORE_PTX_CALL(_L1_REUSE, _L2_HINT, ...)                                      \
    if constexpr (_L1_REUSE == L1_unchanged_reuse)                                            \
    {                                                                                         \
      _CUDA_VPTX::st_L1_evict_unchanged##_L2_HINT(_CUDA_VPTX::space_global_t{}, __VA_ARGS__); \
    }                                                                                         \
    else if constexpr (_L1_REUSE == L1_normal_reuse)                                          \
    {                                                                                         \
      _CUDA_VPTX::st_L1_evict_normal##_L2_HINT(_CUDA_VPTX::space_global_t{}, __VA_ARGS__);    \
    }                                                                                         \
    else if constexpr (_L1_REUSE == L1_low_reuse)                                             \
    {                                                                                         \
      _CUDA_VPTX::st_L1_evict_first##_L2_HINT(_CUDA_VPTX::space_global_t{}, __VA_ARGS__);     \
    }                                                                                         \
    else if constexpr (_L1_REUSE == L1_high_reuse)                                            \
    {                                                                                         \
      _CUDA_VPTX::st_L1_evict_last##_L2_HINT(_CUDA_VPTX::space_global_t{}, __VA_ARGS__);      \
    }                                                                                         \
    else if constexpr (_L1_REUSE == L1_no_reuse)                                              \
    {                                                                                         \
      _CUDA_VPTX::st_L1_no_allocate##_L2_HINT(_CUDA_VPTX::space_global_t{}, __VA_ARGS__);     \
    }

/***********************************************************************************************************************
 * SM-Specific Functions
 **********************************************************************************************************************/

template <typename _Tp, _L1_ReuseEnum _Ep, typename _AccessProperty>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __store_sm80(
  _Tp __data, _Tp* __ptr, __l1_reuse_t<_Ep> __l1_reuse, [[maybe_unused]] __l2_hint_t<_AccessProperty> __l2_hint) noexcept
{
  if constexpr (!__l2_hint)
  {
    _CCCL_STORE_PTX_CALL(__l1_reuse, , __ptr, __data);
  }
  else
  {
    _CCCL_STORE_PTX_CALL(__l1_reuse, _L2_cache_hint, __ptr, __data, __l2_hint.__property);
  }
}

template <typename _Tp, _L1_ReuseEnum _Ep>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __store_sm70(_Tp __data, _Tp* __ptr, __l1_reuse_t<_Ep> __l1_reuse) noexcept
{
  _CCCL_STORE_PTX_CALL(__l1_reuse, , __ptr, __data);
}

/***********************************************************************************************************************
 * INTERNAL DISPATCH
 **********************************************************************************************************************/

template <typename _Tp, _L1_ReuseEnum _Ep, typename _AccessProperty>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __store_dispatch(
  _Tp __data,
  _Tp* __ptr,
  [[maybe_unused]] __l1_reuse_t<_Ep> __l1_reuse,
  [[maybe_unused]] __l2_hint_t<_AccessProperty> __l2_hint) noexcept
{
  static_assert(sizeof(_Tp) <= __max_ptx_access_size);
  // clang-format off
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_80, (_CUDA_VDEV::__store_sm80(__data, __ptr, __l1_reuse, __l2_hint);),
                     NV_PROVIDES_SM_70, (_CUDA_VDEV::__store_sm70(__data, __ptr, __l1_reuse);),
                     NV_IS_DEVICE,      (*__ptr = __data;)); // fallback
  // clang-format on
}

/***********************************************************************************************************************
 * UTILITIES
 **********************************************************************************************************************/

template <typename _Tp, size_t _Np, _L1_ReuseEnum _Ep, typename _AccessProperty, size_t... _Ip>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __unroll_store(
  const _CUDA_VSTD::array<_Tp, _Np>& data,
  _Tp* __ptr,
  __l1_reuse_t<_Ep> __l1_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint,
  _CUDA_VSTD::index_sequence<_Ip...> = {})
{
  if constexpr (__l1_reuse == L1_unchanged_reuse && !__l2_hint)
  {
    ((__ptr[_Ip] = data[_Ip]), ...);
  }
  else
  {
    ((_CUDA_VDEV::__store_dispatch(data[_Ip], __ptr + _Ip, __l1_reuse, __l2_hint)), ...);
  }
};

/***********************************************************************************************************************
 * INTERNAL API
 **********************************************************************************************************************/

template <typename _Tp, _L1_ReuseEnum _Ep, typename _AccessProperty>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
__store_element(_Tp __data, _Tp* __ptr, __l1_reuse_t<_Ep> __l1_reuse, __l2_hint_t<_AccessProperty> __l2_hint) noexcept
{
  static_assert(!_CUDA_VSTD::is_const_v<_Tp>, "_Tp must not be const");
  _CCCL_ASSERT(__ptr != nullptr, "'ptr' must not be null");
  _CCCL_ASSERT(__isGlobal(__ptr), "'ptr' must point to global memory");
  _CCCL_ASSERT(_CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % alignof(_Tp) == 0, "'ptr' must be aligned");
  auto __ptr_gmem = _CUDA_VSTD::bit_cast<_Tp*>(__cvta_generic_to_global(__ptr));
  if constexpr (__l1_reuse == L1_unchanged_reuse && !__l2_hint)
  {
    *__ptr_gmem = __data;
  }
  else
  {
    static_assert(_CUDA_VSTD::has_single_bit(sizeof(_Tp)) || sizeof(_Tp) % __max_ptx_access_size == 0,
                  "'sizeof(_Tp)' must be a power of 2 or multiple of max_alignment with non-default properties");
    constexpr auto __access_size = _CUDA_VSTD::min(sizeof(_Tp), __max_ptx_access_size);
    constexpr auto __num_unroll  = sizeof(_Tp) / __access_size;
    using __aligned_data         = _AlignedData<__access_size>;
    using __store_type           = _CUDA_VSTD::array<__aligned_data, __num_unroll>;
    auto __index_seq             = _CUDA_VSTD::make_index_sequence<__num_unroll>{};
    auto __ptr_gmem2             = reinterpret_cast<__aligned_data*>(__ptr_gmem);
    auto __data_tmp              = _CUDA_VSTD::bit_cast<__store_type>(__data);
    _CUDA_VDEV::__unroll_store(__data_tmp, __ptr_gmem2, __l1_reuse, __l2_hint, __index_seq);
  }
}

template <size_t _Np, typename _Tp, size_t _Align, _L1_ReuseEnum _Ep, typename _AccessProperty>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __store_array(
  _CUDA_VSTD::array<_Tp, _Np> __data,
  _Tp* __ptr,
  aligned_size_t<_Align>,
  __l1_reuse_t<_Ep> __l1_reuse,
  __l2_hint_t<_AccessProperty> __l2_hint) noexcept
{
  constexpr auto __access_size = sizeof(_Tp) * _Np;
  static_assert(!_CUDA_VSTD::is_const_v<_Tp>, "_Tp must not be const");
  static_assert(_Np > 0);
  static_assert(_CUDA_VSTD::has_single_bit(_Align), "_Align must be a power of 2");
  static_assert(_Align >= alignof(_Tp), "_Align must be greater than or equal to alignof(_Tp)");
  static_assert(__access_size % _Align == 0, "Np * sizeof(_Tp) must be a multiple of _Align");
  _CCCL_ASSERT(__ptr != nullptr, "'ptr' must not be null");
  _CCCL_ASSERT(__isGlobal(__ptr), "'ptr' must point to global memory");
  _CCCL_ASSERT(_CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % _Align == 0, "'ptr' must be aligned");
  constexpr bool __is_default_access = __l1_reuse == L1_unchanged_reuse && !__l2_hint;
  constexpr auto __max_align         = __is_default_access ? _Align : _CUDA_VSTD::min(_Align, __max_ptx_access_size);
  constexpr auto __num_unroll        = __access_size / __max_align;
  using __store_type                 = _AlignedData<__max_align>;
  using __store_array_type           = _CUDA_VSTD::array<__store_type, __num_unroll>;
  auto __ptr_gmem                    = _CUDA_VSTD::bit_cast<__store_type*>(__cvta_generic_to_global(__ptr));
  auto __tmp                         = _CUDA_VSTD::bit_cast<__store_array_type>(__data);
  auto __index_seq                   = _CUDA_VSTD::make_index_sequence<__num_unroll>{};
  _CUDA_VDEV::__unroll_store(__tmp, __ptr_gmem, __l1_reuse, __l2_hint, __index_seq);
}

/***********************************************************************************************************************
 * USER API
 **********************************************************************************************************************/

template <typename _Tp, _L1_ReuseEnum _Ep = _L1_ReuseEnum::_Unchanged, typename _AccessProperty = access_property::global>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(_Tp __data,
      _Tp* __ptr,
      __l1_reuse_t<_Ep> __l1_reuse         = L1_unchanged_reuse,
      _AccessProperty __l2_access_property = access_property::global{}) noexcept
{
  _CUDA_VDEV::__store_element(__data, __ptr, __l1_reuse, __l2_hint_t{__l2_access_property});
}

template <typename _Tp, typename _Prop, _L1_ReuseEnum _Ep = _L1_ReuseEnum::_Unchanged>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(_Tp __data, annotated_ptr<_Tp, _Prop> __ptr, __l1_reuse_t<_Ep> __l1_reuse = L1_unchanged_reuse) noexcept
{
  _CUDA_VDEV::__store_element(__data, __ptr.__get_raw_ptr(), __l1_reuse, __ptr.__property());
}

template <size_t _Np,
          typename _Tp,
          size_t _Align            = alignof(_Tp),
          _L1_ReuseEnum _Ep        = _L1_ReuseEnum::_Unchanged,
          typename _AccessProperty = access_property::global>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(const _CUDA_VSTD::array<_Tp, _Np>& __data,
      _Tp* __ptr,
      aligned_size_t<_Align> __align       = aligned_size_t<_Align>{alignof(_Tp)},
      __l1_reuse_t<_Ep> __l1_reuse         = L1_unchanged_reuse,
      _AccessProperty __l2_access_property = access_property::global{}) noexcept
{
  _CUDA_VDEV::__store_array<_Np>(__data, __ptr, __align, __l1_reuse, __l2_hint_t{__l2_access_property});
}

template <size_t _Np,
          typename _Tp,
          typename _Prop,
          size_t _Align            = alignof(_Tp),
          _L1_ReuseEnum _Ep        = _L1_ReuseEnum::_Unchanged,
          typename _AccessProperty = access_property::global>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(const _CUDA_VSTD::array<_Tp, _Np>& __data,
      annotated_ptr<_Tp, _Prop> __ptr,
      aligned_size_t<_Align> __align = aligned_size_t<_Align>{alignof(_Tp)},
      __l1_reuse_t<_Ep> __l1_reuse   = L1_unchanged_reuse) noexcept
{
  _CUDA_VDEV::__store_array<_Np>(__data, __ptr.__get_raw_ptr(), __align, __l1_reuse, __ptr.__property());
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_HAS_CUDA_COMPILER
#endif // _CUDA___DATA_MOVEMENT_STORE_H
