//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_SCALAR_IMPL_H
#define _CUDAX___SIMD_SCALAR_IMPL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__simd/declaration.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
namespace simd_abi
{
struct __scalar
{
  static constexpr ::cuda::std::size_t __simd_size = 1;
};
} // namespace simd_abi

template <typename _Tp>
struct __simd_storage<_Tp, simd_abi::__scalar>
{
  using value_type = _Tp;
  _Tp __data;

  [[nodiscard]] _CCCL_API constexpr _Tp __get([[maybe_unused]] ::cuda::std::size_t __idx) const noexcept
  {
    _CCCL_ASSERT(__idx == 0, "Index is out of bounds");
    return __data;
  }

  _CCCL_API constexpr void __set([[maybe_unused]] ::cuda::std::size_t __idx, _Tp __v) noexcept
  {
    _CCCL_ASSERT(__idx == 0, "Index is out of bounds");
    __data = __v;
  }
};

template <typename _Tp>
struct __mask_storage<_Tp, simd_abi::__scalar> : __simd_storage<bool, simd_abi::__scalar>
{
  using value_type = bool;
};

// *********************************************************************************************************************
// * SIMD Arithmetic Operations
// *********************************************************************************************************************

template <typename _Tp>
struct __simd_operations<_Tp, simd_abi::__scalar>
{
  using _SimdStorage = __simd_storage<_Tp, simd_abi::__scalar>;
  using _MaskStorage = __mask_storage<_Tp, simd_abi::__scalar>;

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __broadcast(_Tp __v) noexcept
  {
    return _SimdStorage{__v};
  }

  template <typename _Generator>
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __generate(_Generator&& __g)
  {
    return _SimdStorage{__g(::cuda::std::integral_constant<::cuda::std::size_t, 0>())};
  }

  template <typename _Up>
  _CCCL_API static constexpr void __load(_SimdStorage& __s, const _Up* __mem) noexcept
  {
    __s.__data = static_cast<_Tp>(__mem[0]);
  }

  template <typename _Up>
  _CCCL_API static constexpr void __store(const _SimdStorage& __s, _Up* __mem) noexcept
  {
    __mem[0] = static_cast<_Up>(__s.__data);
  }

  _CCCL_API static constexpr void __increment(_SimdStorage& __s) noexcept
  {
    __s.__data += 1;
  }

  _CCCL_API static constexpr void __decrement(_SimdStorage& __s) noexcept
  {
    __s.__data -= 1;
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __negate(const _SimdStorage& __s) noexcept
  {
    return _MaskStorage{!__s.__data};
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __bitwise_not(const _SimdStorage& __s) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(~__s.__data)};
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __unary_minus(const _SimdStorage& __s) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(-__s.__data)};
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __plus(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(__lhs.__data + __rhs.__data)};
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __minus(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(__lhs.__data - __rhs.__data)};
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __multiplies(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(__lhs.__data * __rhs.__data)};
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __divides(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(__lhs.__data / __rhs.__data)};
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __equal_to(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _MaskStorage{__lhs.__data == __rhs.__data};
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __not_equal_to(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _MaskStorage{__lhs.__data != __rhs.__data};
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __less(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _MaskStorage{__lhs.__data < __rhs.__data};
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __less_equal(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _MaskStorage{__lhs.__data <= __rhs.__data};
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __greater(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _MaskStorage{__lhs.__data > __rhs.__data};
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __greater_equal(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _MaskStorage{__lhs.__data >= __rhs.__data};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __modulo(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(__lhs.__data % __rhs.__data)};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_and(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(__lhs.__data & __rhs.__data)};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_or(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(__lhs.__data | __rhs.__data)};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_xor(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(__lhs.__data ^ __rhs.__data)};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __shift_left(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(__lhs.__data << __rhs.__data)};
  }

  _CCCL_TEMPLATE(typename _Up = _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __shift_right(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    return _SimdStorage{static_cast<_Tp>(__lhs.__data >> __rhs.__data)};
  }
};

// *********************************************************************************************************************
// * SIMD Mask Operations
// *********************************************************************************************************************

template <class _Tp>
struct __mask_operations<_Tp, simd_abi::__scalar>
{
  using _MaskStorage = __mask_storage<_Tp, simd_abi::__scalar>;

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __broadcast(bool __v) noexcept
  {
    return _MaskStorage{__v};
  }

  template <typename _Generator>
  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __generate(_Generator&& __g)
  {
    return _MaskStorage{static_cast<bool>(__g(::cuda::std::integral_constant<::cuda::std::size_t, 0>()))};
  }

  _CCCL_API static constexpr void __load(_MaskStorage& __s, const bool* __mem) noexcept
  {
    __s.__data = __mem[0];
  }

  _CCCL_API static constexpr void __store(const _MaskStorage& __s, bool* __mem) noexcept
  {
    __mem[0] = static_cast<bool>(__s.__data);
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __bitwise_and(const _MaskStorage& __lhs, const _MaskStorage& __rhs) noexcept
  {
    return _MaskStorage{__lhs.__data && __rhs.__data};
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __bitwise_or(const _MaskStorage& __lhs, const _MaskStorage& __rhs) noexcept
  {
    return _MaskStorage{__lhs.__data || __rhs.__data};
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage
  __bitwise_xor(const _MaskStorage& __lhs, const _MaskStorage& __rhs) noexcept
  {
    return _MaskStorage{__lhs.__data != __rhs.__data};
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __bitwise_not(const _MaskStorage& __s) noexcept
  {
    return _MaskStorage{!__s.__data};
  }

  [[nodiscard]] _CCCL_API static constexpr bool __equal_to(const _MaskStorage& __lhs, const _MaskStorage& __rhs) noexcept
  {
    return __lhs.__data == __rhs.__data;
  }

  [[nodiscard]] _CCCL_API static constexpr bool __all(const _MaskStorage& __s) noexcept
  {
    return static_cast<bool>(__s.__data);
  }

  [[nodiscard]] _CCCL_API static constexpr bool __any(const _MaskStorage& __s) noexcept
  {
    return static_cast<bool>(__s.__data);
  }

  [[nodiscard]] _CCCL_API static constexpr int __count(const _MaskStorage& __s) noexcept
  {
    return static_cast<bool>(__s.__data) ? 1 : 0;
  }
};
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_SCALAR_IMPL_H
