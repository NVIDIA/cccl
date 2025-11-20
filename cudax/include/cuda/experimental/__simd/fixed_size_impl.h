//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_FIXED_SIZE_IMPL_H
#define _CUDAX___SIMD_FIXED_SIZE_IMPL_H

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
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__simd/declaration.h>
#include <cuda/experimental/__simd/utility.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
namespace simd_abi
{
template <int _Np>
struct __fixed_size
{
  static constexpr ::cuda::std::size_t __simd_size = _Np;
};
} // namespace simd_abi

template <typename _Tp, int _Np>
struct __simd_storage<_Tp, simd_abi::__fixed_size<_Np>>
{
  _Tp __data[_Np];

  [[nodiscard]] _CCCL_API constexpr _Tp __get([[maybe_unused]] ::cuda::std::size_t __idx) const noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__idx, 0, __simd_size), "Index is out of bounds");
    return __data[__idx];
  }

  _CCCL_API constexpr void __set([[maybe_unused]] ::cuda::std::size_t __idx, _Tp __v) noexcept
  {
    _CCCL_ASSERT(::cuda::in_range(__idx, 0, __simd_size), "Index is out of bounds");
    __data[__idx] = __v;
  }
};

template <typename _Tp, int _Np>
struct __mask_storage<_Tp, simd_abi::__fixed_size<_Np>>
    : __simd_storage<::cuda::std::__make_nbit_uint_t<::cuda::std::__num_bits_v<_Tp>>, simd_abi::__fixed_size<_Np>>
{};

// *********************************************************************************************************************
// * SIMD Arithmetic Operations
// *********************************************************************************************************************

template <typename _Tp, int _Np>
struct __simd_operations<_Tp, simd_abi::__fixed_size<_Np>>
{
  using _SimdStorage _CCCL_NODEBUG = __simd_storage<_Tp, simd_abi::__fixed_size<_Np>>;
  using _MaskStorage _CCCL_NODEBUG = __mask_storage<_Tp, simd_abi::__fixed_size<_Np>>;

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __broadcast(_Tp __v) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; ++__i)
    {
      __result.__data[__i] = __v;
    }
    return __result;
  }

  template <typename _Generator, ::cuda::std::size_t... _Is>
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __generate_init(_Generator&& __g, ::cuda::std::index_sequence<_Is...>)
  {
    return _SimdStorage{{__g(std::integral_constant<::cuda::std::size_t, _Is>())...}};
  }

  template <typename _Generator>
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __generate(_Generator&& __g)
  {
    return __generate_init(::cuda::std::forward<_Generator>(__g), ::cuda::std::make_index_sequence<_Np>());
  }

  template <typename _Up>
  _CCCL_API static constexpr void __load(_SimdStorage& __s, const _Up* __mem) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __s.__data[__i] = static_cast<_Tp>(__mem[__i]);
    }
  }

  template <typename _Up>
  _CCCL_API static constexpr void __store(const _SimdStorage& __s, _Up* __mem) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __mem[__i] = static_cast<_Up>(__s.__data[__i]);
    }
  }

  _CCCL_API static constexpr void __increment(_SimdStorage& __s) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __s.__data[__i] += 1;
    }
  }

  _CCCL_API static constexpr void __decrement(_SimdStorage& __s) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __s.__data[__i] -= 1;
    }
  }

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __negate(const _SimdStorage& __s) noexcept
  {
    _MaskStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = !__s.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __bitwise_not(const _SimdStorage& __s) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = ~__s.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage __unary_minus(const _SimdStorage& __s) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = -__s.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __plus(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] + __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __minus(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] - __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __multiplies(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] * __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __divides(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] / __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __equal_to(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] == __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __not_equal_to(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] != __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __less(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] < __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __less_equal(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] <= __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __greater(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] > __rhs.__data[__i];
    }
    return __result;
  }

  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __greater_equal(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] >= __rhs.__data[__i];
    }
    return __result;
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __modulo(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] % __rhs.__data[__i];
    }
    return __result;
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_and(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] & __rhs.__data[__i];
    }
    return __result;
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_or(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] | __rhs.__data[__i];
    }
    return __result;
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __bitwise_xor(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] ^ __rhs.__data[__i];
    }
    return __result;
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __shift_left(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] << __rhs.__data[__i];
    }
    return __result;
  }

  _CCCL_TEMPLATE(typename _Up)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Up>)
  [[nodiscard]] _CCCL_API static constexpr _SimdStorage
  __shift_right(const _SimdStorage& __lhs, const _SimdStorage& __rhs) noexcept
  {
    _SimdStorage __result;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __result.__data[__i] = __lhs.__data[__i] >> __rhs.__data[__i];
    }
    return __result;
  }
};

// *********************************************************************************************************************
// * SIMD Mask Operations
// *********************************************************************************************************************

template <class _Tp, int _Np>
struct __mask_operations<_Tp, simd_abi::__fixed_size<_Np>>
{
  using _MaskStorage = __mask_storage<_Tp, simd_abi::__fixed_size<_Np>>;

  [[nodiscard]] _CCCL_API static constexpr _MaskStorage __broadcast(bool __v) noexcept
  {
    _MaskStorage __result;
    const auto __all_bits_v = ::cuda::experimental::datapar::__set_all_bits<_Tp>(__v);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; ++__i)
    {
      __result.__set(__i, __all_bits_v);
    }
    return __result;
  }

  _CCCL_API static constexpr void __load(_MaskStorage& __s, const bool* __mem) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __s.__data[__i] = ::cuda::experimental::datapar::__set_all_bits<_Tp>(__mem[__i]);
    }
  }

  _CCCL_API static constexpr void __store(const _MaskStorage& __s, bool* __mem) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Np; __i++)
    {
      __mem[__i] = static_cast<bool>(__s.__data[__i]);
    }
  }
};
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_FIXED_SIZE_IMPL_H
