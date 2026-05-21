//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_SIMD_INTRINSICS_ARRAY_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_SIMD_INTRINSICS_ARRAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/__simd/specializations/fixed_size_storage.h>
#include <cuda/std/__simd/specializations/simd_intrinsics.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------
// conversion utilities

template <typename _SimdStorage>
inline constexpr size_t __simd_storage_size_u32 = 0;

template <typename _Tp, __simd_size_type _Np>
inline constexpr size_t __simd_storage_size_u32<__simd_storage<_Tp, __fixed_size<_Np>>> =
  ::cuda::ceil_div(_Np, sizeof(uint32_t) / sizeof(_Tp));

template <size_t _Np>
using __array_u32_t = array<uint32_t, _Np>;

template <typename _SimdStorage>
using __simd_storage_u32_t = __array_u32_t<__simd_storage_size_u32<_SimdStorage>>;

template <typename _SimdStorage>
inline constexpr size_t __simd_storage_copy_size_u32 = 0;

template <typename _Tp, __simd_size_type _Np>
inline constexpr size_t __simd_storage_copy_size_u32<__simd_storage<_Tp, __fixed_size<_Np>>> = _Np * sizeof(_Tp);

template <typename _SimdStorage, typename _SimdStorageU32 = __simd_storage_u32_t<_SimdStorage>>
[[nodiscard]] _CCCL_API constexpr _SimdStorageU32 __to_unsigned_storage(const _SimdStorage& __s) noexcept
{
  _SimdStorageU32 __tmp{};
  const auto __input_data = ::cuda::std::assume_aligned<alignof(uint32_t)>(__s.__data);
  ::cuda::std::memcpy(__tmp.data(), __input_data, __simd_storage_copy_size_u32<_SimdStorage>);
  return __tmp;
}

template <typename _SimdStorage, typename _SimdStorageU32 = __simd_storage_u32_t<_SimdStorage>>
[[nodiscard]] _CCCL_API constexpr _SimdStorage __copy_from_unsigned_storage(const _SimdStorageU32& __tmp) noexcept
{
  _SimdStorage __result{};
  const auto __result_ptr = ::cuda::std::assume_aligned<alignof(uint32_t)>(__result.__data);
  ::cuda::std::memcpy(__result_ptr, __tmp.data(), __simd_storage_copy_size_u32<_SimdStorage>);
  return __result;
}

//----------------------------------------------------------------------------------------------------------------------
// device-only functions

#if _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()

template <typename _Tp, size_t _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr __array_u32_t<_Np>
__vadd_16bit_x2(const __array_u32_t<_Np>& __lhs_u, const __array_u32_t<_Np>& __rhs_u) noexcept
{
  __array_u32_t<_Np> __result_u;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (size_t __i = 0; __i < _Np; ++__i)
  {
    if constexpr (is_unsigned_v<_Tp>)
    {
      __result_u[__i] = ::cuda::std::simd::__vadd_u16x2(__lhs_u[__i], __rhs_u[__i]);
    }
    else
    {
      __result_u[__i] = ::cuda::std::simd::__vadd_s16x2(__lhs_u[__i], __rhs_u[__i]);
    }
  }
  return __result_u;
}

#  if _CCCL_HAS_SIMD_8BIT()

template <typename _Tp, size_t _Np>
[[nodiscard]] _CCCL_DEVICE_API constexpr __array_u32_t<_Np>
__vadd_8bit_x4(const __array_u32_t<_Np>& __lhs_u, const __array_u32_t<_Np>& __rhs_u) noexcept
{
  __array_u32_t<_Np> __result_u;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (size_t __i = 0; __i < _Np; ++__i)
  {
    if constexpr (is_unsigned_v<_Tp>)
    {
      __result_u[__i] = ::cuda::std::simd::__vadd_u8x4(__lhs_u[__i], __rhs_u[__i]);
    }
    else
    {
      __result_u[__i] = ::cuda::std::simd::__vadd_s8x4(__lhs_u[__i], __rhs_u[__i]);
    }
  }
  return __result_u;
}

#  endif // _CCCL_HAS_SIMD_8BIT()

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_SIMD_INTRINSICS_ARRAY_H
