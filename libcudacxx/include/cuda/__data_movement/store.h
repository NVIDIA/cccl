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

#if _CCCL_HAS_CUDA_COMPILER

#  include <cuda/__barrier/aligned_size.h>
#  include <cuda/__data_movement/aligned_data.h>
#  include <cuda/__data_movement/properties.h>
#  include <cuda/__ptx/instructions/st.h>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__bit/has_single_bit.h>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/span>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

/***********************************************************************************************************************
 * INTERNAL LOAD FUNCTIONS
 **********************************************************************************************************************/

template <typename _Tp, _EvictionPolicyEnum _Ep>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
__store_sm70(_Tp __data, _Tp* __ptr, __eviction_policy_t<_Ep> __eviction_policy) noexcept
{
  if constexpr (__eviction_policy == eviction_unchanged)
  {
    _CUDA_VPTX::st_global_L1_evict_unchanged(__ptr, __data);
  }
  else if constexpr (__eviction_policy == eviction_normal)
  {
    _CUDA_VPTX::st_global_L1_evict_normal(__ptr, __data);
  }
  else if constexpr (__eviction_policy == eviction_first)
  {
    _CUDA_VPTX::st_global_L1_evict_first(__ptr, __data);
  }
  else if constexpr (__eviction_policy == eviction_last)
  {
    _CUDA_VPTX::st_global_L1_evict_last(__ptr, __data);
  }
  else if constexpr (__eviction_policy == eviction_no_alloc)
  {
    _CUDA_VPTX::st_global_L1_no_allocate(__ptr, __data);
  }
}

template <typename _Tp, _EvictionPolicyEnum _Ep>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
__store_dispatch(_Tp __data, _Tp* __ptr, [[maybe_unused]] __eviction_policy_t<_Ep> __eviction_policy) noexcept
{
  static_assert(sizeof(_Tp) <= 16);
#  if __cccl_ptx_isa >= 830
  // clang-format off
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_70, (return _CUDA_VDEV::__store_sm70(__data, __ptr, __eviction_policy);),
                     NV_IS_DEVICE,      (*__ptr = __data;)); // fallback
  // clang-format on
#  elif __cccl_ptx_isa >= 740 // __cccl_ptx_isa >= 740 && __cccl_ptx_isa < 830
  if constexpr (sizeof(_Tp) <= 8)
  {
    // clang-format off
    NV_DISPATCH_TARGET(NV_PROVIDES_SM_70, (return _CUDA_VDEV::__store_sm70(__data, __ptr, __eviction_policy);),
                       NV_IS_DEVICE,      (*__ptr = __data;)); // fallback
    // clang-format on
  }
  else
  {
    *__ptr = __data;
  }
#  else // __cccl_ptx_isa < 740
  *__ptr = __data;
#  endif
}

/***********************************************************************************************************************
 * UTILITIES
 **********************************************************************************************************************/

template <typename _Tp, size_t _Np, _EvictionPolicyEnum _Ep, size_t... _Ip>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __unroll_store(
  const _CUDA_VSTD::array<_Tp, _Np>& data,
  _Tp* __ptr,
  __eviction_policy_t<_Ep> __eviction_policy,
  _CUDA_VSTD::index_sequence<_Ip...> = {})
{
  if constexpr (__eviction_policy == eviction_none)
  {
    ((__ptr[_Ip] = data[_Ip]), ...);
  }
  else
  {
    ((_CUDA_VDEV::__store_dispatch(data[_Ip], __ptr + _Ip, __eviction_policy)), ...);
  }
};

/***********************************************************************************************************************
 * USER API
 **********************************************************************************************************************/

template <typename _Tp, _EvictionPolicyEnum _Ep = _EvictionPolicyEnum::_None>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(_Tp __data, _Tp* __ptr, __eviction_policy_t<_Ep> __eviction_policy = eviction_none) noexcept
{
  _CCCL_ASSERT(__ptr != nullptr, "'ptr' must not be null");
  _CCCL_ASSERT(_CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % alignof(_Tp) == 0, "'ptr' must be aligned");
  _CCCL_ASSERT(__isGlobal(__ptr), "'ptr' must point to global memory");
  auto __ptr_gmem = _CUDA_VSTD::bit_cast<_Tp*>(__cvta_generic_to_global(__ptr));
  if constexpr (__eviction_policy == eviction_none)
  {
    *__ptr_gmem = __data;
  }
  else
  {
    static_assert(_CUDA_VSTD::has_single_bit(sizeof(_Tp)),
                  "'sizeof(_Tp)' must be a power of 2 with non-default properties");
    if constexpr (sizeof(_Tp) > 16)
    {
      constexpr auto __num_16_bytes = sizeof(_Tp) / 16;
      using __bytes_16              = _AlignedData<16>;
      using __store_type            = _CUDA_VSTD::array<__bytes_16, __num_16_bytes>;
      auto __index_seq              = _CUDA_VSTD::make_index_sequence<__num_16_bytes>{};
      auto __ptr2                   = reinterpret_cast<__bytes_16*>(__ptr_gmem);
      auto __data_tmp               = _CUDA_VSTD::bit_cast<__store_type>(__data);
      _CUDA_VDEV::__unroll_store(__data_tmp, __ptr2, __eviction_policy, __index_seq);
    }
    else
    {
      return _CUDA_VDEV::__store_dispatch(__data, __ptr_gmem, __eviction_policy);
    }
  }
}

template <size_t _Np, typename _Tp, size_t _Align = alignof(_Tp), _EvictionPolicyEnum _Ep = _EvictionPolicyEnum::_None>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(_CUDA_VSTD::array<_Tp, _Np> __data,
      _Tp* __ptr,
      ::cuda::aligned_size_t<_Align>             = ::cuda::aligned_size_t<_Align>{alignof(_Tp)},
      __eviction_policy_t<_Ep> __eviction_policy = eviction_none) noexcept
{
  static_assert(!_CUDA_VSTD::is_const_v<_Tp>, "_Tp must not be const");
  static_assert(_Np > 0);
  static_assert(_CUDA_VSTD::has_single_bit(_Align), "_Align must be a power of 2");
  static_assert(_Align >= alignof(_Tp), "_Align must be greater than or equal to alignof(_Tp)");
  static_assert(sizeof(_Tp) * _Np % _Align == 0, "Np * sizeof(_Tp) must be a multiple of _Align");
  _CCCL_ASSERT(__ptr != nullptr, "'ptr' must not be null");
  _CCCL_ASSERT(_CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % _Align == 0, "'ptr' must be aligned");
  _CCCL_ASSERT(__isGlobal(__ptr), "'ptr' must point to global memory");
  constexpr bool __is_default_access = __eviction_policy == eviction_none;
  constexpr auto __max_align         = __is_default_access ? _Align : _CUDA_VSTD::min(_Align, size_t{16});
  constexpr auto __count             = (sizeof(_Tp) * _Np) / __max_align;
  using __store_type                 = _AlignedData<__max_align>;
  using __pointer_type               = _CUDA_VSTD::array<__store_type, __count>;
  auto __ptr_gmem                    = _CUDA_VSTD::bit_cast<__store_type*>(__cvta_generic_to_global(__ptr));
  auto __tmp                         = _CUDA_VSTD::bit_cast<__pointer_type>(__data);
  auto __index_seq                   = _CUDA_VSTD::make_index_sequence<__count>{};
  _CUDA_VDEV::__unroll_store(__tmp, __ptr_gmem, __eviction_policy, __index_seq);
}

template <size_t _Np, typename _Tp, size_t _Align = alignof(_Tp), _EvictionPolicyEnum _Ep = _EvictionPolicyEnum::_None>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(_CUDA_VSTD::span<_Tp, _Np> __data,
      _Tp* __ptr,
      ::cuda::aligned_size_t<_Align> __align     = ::cuda::aligned_size_t<_Align>{alignof(_Tp)},
      __eviction_policy_t<_Ep> __eviction_policy = eviction_none) noexcept
{
  static_assert(_Np > 0 && _Np != _CUDA_VSTD::dynamic_extent);
  _CUDA_VSTD::array<_Tp, _Np> __tmp;
#  pragma unroll
  for (size_t i = 0; i < _Np; ++i)
  {
    __tmp[i] = __data[i];
  }
  _CUDA_VDEV::store(__tmp, __ptr, __align, __eviction_policy);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_HAS_CUDA_COMPILER
#endif // _CUDA___DATA_MOVEMENT_STORE_H
