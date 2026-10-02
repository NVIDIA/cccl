// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * Common type manipulation (metaprogramming) utilities
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/align_bytes.cuh> // IWYU pragma: export
#include <cub/detail/binary_op_has_idx_param.cuh> // IWYU pragma: export
#include <cub/detail/constant.cuh> // IWYU pragma: export
#include <cub/detail/cub_vector.cuh> // IWYU pragma: export
#include <cub/detail/detect_nested_type.cuh> // IWYU pragma: export
#include <cub/detail/double_buffer.cuh> // IWYU pragma: export
#include <cub/detail/future_value.cuh> // IWYU pragma: export
#include <cub/detail/input_value.cuh> // IWYU pragma: export
#include <cub/detail/it_traits.cuh> // IWYU pragma: export
#include <cub/detail/key_value_pair.cuh> // IWYU pragma: export
#include <cub/detail/lazy_trait.cuh> // IWYU pragma: export
#include <cub/detail/log2.cuh> // IWYU pragma: export
#include <cub/detail/non_void_value.cuh> // IWYU pragma: export
#include <cub/detail/null_type.cuh> // IWYU pragma: export
#include <cub/detail/power_of_two.cuh> // IWYU pragma: export
#include <cub/detail/type_size.cuh> // IWYU pragma: export
#include <cub/detail/type_traits.cuh>
#include <cub/detail/uninitialized.cuh> // IWYU pragma: export
#include <cub/detail/uninitialized_copy.cuh>
#include <cub/detail/unit_word.cuh> // IWYU pragma: export

#include <thrust/iterator/detail/any_assign.h>

#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

{

/******************************************************************************
 * Marker types
 ******************************************************************************/

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

/******************************************************************************
 * Simple type traits utilities.
 ******************************************************************************/

/**
 * \brief Basic type traits categories
 */
enum Category // NOLINT(cppcoreguidelines-use-enum-class)
{
  NOT_A_NUMBER,
  SIGNED_INTEGER,
  UNSIGNED_INTEGER,
  FLOATING_POINT
};

namespace detail
{
struct is_primitive_impl;

// case for Kind = NOT_A_NUMBER, or Primitive = false
template <Category Kind, bool Primitive, typename UnsignedBitsT, typename T>
struct BaseTraits
{
private:
  friend struct is_primitive_impl;

  static constexpr bool is_primitive = Primitive;
};

template <typename UnsignedBitsT, typename T>
struct BaseTraits<UNSIGNED_INTEGER, true, UnsignedBitsT, T>
{
  static_assert(sizeof(UnsignedBitsT) == sizeof(T),
                "The size of the unsigned type holding the bits of T must be the same as T");
  static_assert(::cuda::std::numeric_limits<T>::is_specialized,
                "Please also specialize cuda::std::numeric_limits for T");

  using UnsignedBits                       = UnsignedBitsT;
  static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(0);
  static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1);

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
  {
    return key;
  }

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleOut(UnsignedBits key)
  {
    return key;
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::max()") static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Max()
  {
    UnsignedBits retval_bits = MAX_KEY;
    T retval;
    memcpy(&retval, &retval_bits, sizeof(T));
    return retval;
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::lowest()") static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T
  Lowest()
  {
    UnsignedBits retval_bits = LOWEST_KEY;
    T retval;
    memcpy(&retval, &retval_bits, sizeof(T));
    return retval;
  }

private:
  friend struct is_primitive_impl;

  static constexpr bool is_primitive = true;
};

template <typename UnsignedBitsT, typename T>
struct BaseTraits<SIGNED_INTEGER, true, UnsignedBitsT, T>
{
  static_assert(sizeof(UnsignedBitsT) == sizeof(T),
                "The size of the unsigned type holding the bits of T must be the same as T");
  static_assert(::cuda::std::numeric_limits<T>::is_specialized,
                "Please also specialize cuda::std::numeric_limits for T");

  using UnsignedBits = UnsignedBitsT;

  static constexpr UnsignedBits HIGH_BIT   = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
  static constexpr UnsignedBits LOWEST_KEY = HIGH_BIT;
  static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
  {
    return key ^ HIGH_BIT;
  };

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleOut(UnsignedBits key)
  {
    return key ^ HIGH_BIT;
  };

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::max()") static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Max()
  {
    UnsignedBits retval = MAX_KEY;
    return reinterpret_cast<T&>(retval);
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::lowest()") static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T
  Lowest()
  {
    UnsignedBits retval = LOWEST_KEY;
    return reinterpret_cast<T&>(retval);
  }

private:
  friend struct is_primitive_impl;

  static constexpr bool is_primitive = true;
};

template <typename UnsignedBitsT, typename T>
struct BaseTraits<FLOATING_POINT, true, UnsignedBitsT, T>
{
  static_assert(sizeof(UnsignedBitsT) == sizeof(T),
                "The size of the unsigned type holding the bits of T must be the same as T");
  static_assert(::cuda::std::numeric_limits<T>::is_specialized,
                "Please also specialize cuda::std::numeric_limits for T");
  static_assert(::cuda::is_floating_point<T>::value, "Please also specialize cuda::is_floating_point for T");
  static_assert(::cuda::is_floating_point_v<T>, "Please also specialize cuda::is_floating_point_v for T");

  using UnsignedBits = UnsignedBitsT;

  static constexpr UnsignedBits HIGH_BIT   = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
  static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(-1);
  static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
  {
    const UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
    return key ^ mask;
  };

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleOut(UnsignedBits key)
  {
    const UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
    return key ^ mask;
  };

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::max()") static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Max()
  {
    return ::cuda::std::numeric_limits<T>::max();
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::lowest()") static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T
  Lowest()
  {
    return ::cuda::std::numeric_limits<T>::lowest();
  }

private:
  friend struct is_primitive_impl;

  static constexpr bool is_primitive = true;
};
} // namespace detail

//! Use this class as base when specializing \ref NumericTraits for primitive signed/unsigned integers or floating-point
//! types.
template <Category Kind, bool Primitive, typename UnsignedBitsT, typename T>
using BaseTraits = detail::BaseTraits<Kind, Primitive, UnsignedBitsT, T>;

//! Numeric type traits for radix sort key operations, decoupled lookback and tuning. You can specialize this template
//! for your own types if:
//! * There is an unsigned integral type of equal size
//! * The size of the type is smaller than 64bits
//! * The arithmetic throughput of the type is similar to other built-in types of the same size
//! For other types, if you want to use them with radix sort, please use the decomposer interface of the radix sort.
// clang-format off
template <typename T, typename = T> struct NumericTraits :            BaseTraits<NOT_A_NUMBER, false, T, T> {};

template <> struct NumericTraits<NullType> :            BaseTraits<NOT_A_NUMBER, false, NullType, NullType> {};

template <> struct NumericTraits<char> :                BaseTraits<(::cuda::std::numeric_limits<char>::is_signed) ? SIGNED_INTEGER : UNSIGNED_INTEGER, true, unsigned char, char> {};
template <> struct NumericTraits<signed char> :         BaseTraits<SIGNED_INTEGER, true, unsigned char, signed char> {};
template <> struct NumericTraits<short> :               BaseTraits<SIGNED_INTEGER, true, unsigned short, short> {};
template <> struct NumericTraits<int> :                 BaseTraits<SIGNED_INTEGER, true, unsigned int, int> {};
template <> struct NumericTraits<long> :                BaseTraits<SIGNED_INTEGER, true, unsigned long, long> {};
template <> struct NumericTraits<long long> :           BaseTraits<SIGNED_INTEGER, true, unsigned long long, long long> {};

template <> struct NumericTraits<unsigned char> :       BaseTraits<UNSIGNED_INTEGER, true, unsigned char, unsigned char> {};
template <> struct NumericTraits<unsigned short> :      BaseTraits<UNSIGNED_INTEGER, true, unsigned short, unsigned short> {};
template <> struct NumericTraits<unsigned int> :        BaseTraits<UNSIGNED_INTEGER, true, unsigned int, unsigned int> {};
template <> struct NumericTraits<unsigned long> :       BaseTraits<UNSIGNED_INTEGER, true, unsigned long, unsigned long> {};
template <> struct NumericTraits<unsigned long long> :  BaseTraits<UNSIGNED_INTEGER, true, unsigned long long, unsigned long long> {};
// clang-format on

#  if _CCCL_HAS_INT128()
template <>
struct NumericTraits<__uint128_t>
{
  using T            = __uint128_t;
  using UnsignedBits = __uint128_t;

  static constexpr UnsignedBits LOWEST_KEY = UnsignedBits(0);
  static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1);

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
  {
    return key;
  }

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleOut(UnsignedBits key)
  {
    return key;
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::max()") static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Max()
  {
    return MAX_KEY;
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::lowest()") static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T
  Lowest()
  {
    return LOWEST_KEY;
  }

private:
  friend struct detail::is_primitive_impl;

  static constexpr bool is_primitive = false;
};

template <>
struct NumericTraits<__int128_t>
{
  using T            = __int128_t;
  using UnsignedBits = __uint128_t;

  static constexpr UnsignedBits HIGH_BIT   = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
  static constexpr UnsignedBits LOWEST_KEY = HIGH_BIT;
  static constexpr UnsignedBits MAX_KEY    = UnsignedBits(-1) ^ HIGH_BIT;

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleIn(UnsignedBits key)
  {
    return key ^ HIGH_BIT;
  };

  static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE UnsignedBits TwiddleOut(UnsignedBits key)
  {
    return key ^ HIGH_BIT;
  };

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::max()") static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T Max()
  {
    UnsignedBits retval = MAX_KEY;
    return reinterpret_cast<T&>(retval);
  }

  //! deprecated [Since 3.0]
  CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits<T>::lowest()") static _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T
  Lowest()
  {
    UnsignedBits retval = LOWEST_KEY;
    return reinterpret_cast<T&>(retval);
  }

private:
  friend struct detail::is_primitive_impl;

  static constexpr bool is_primitive = false;
};
#  endif // _CCCL_HAS_INT128()

// clang-format off
template <> struct NumericTraits<float> :               BaseTraits<FLOATING_POINT, true, unsigned int, float> {};
template <> struct NumericTraits<double> :              BaseTraits<FLOATING_POINT, true, unsigned long long, double> {};
#  if _CCCL_HAS_NVFP16()
    template <typename T> struct NumericTraits<__half, T> :          BaseTraits<FLOATING_POINT, true, unsigned short, T> {};
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
    template <typename T> struct NumericTraits<__nv_bfloat16, T> :   BaseTraits<FLOATING_POINT, true, unsigned short, T> {};
#  endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8()
    template <typename T> struct NumericTraits<__nv_fp8_e4m3, T> :   BaseTraits<FLOATING_POINT, true, unsigned char, T> {};
    template <typename T> struct NumericTraits<__nv_fp8_e5m2, T> :   BaseTraits<FLOATING_POINT, true, unsigned char, T> {};
#endif // _CCCL_HAS_NVFP8()

template <> struct NumericTraits<bool> :                BaseTraits<UNSIGNED_INTEGER, true, typename UnitWord<bool>::VolatileWord, bool> {};
// clang-format on

namespace detail
{
template <typename T>
struct Traits : NumericTraits<::cuda::std::remove_cv_t<T>>
{};
} // namespace detail

//! \brief Query type traits for radix sort key operations, decoupled lookback and tunings. To add support for your own
//! primitive types please specialize \ref NumericTraits.
template <typename T>
using Traits = detail::Traits<T>;

namespace detail
{
// we cannot befriend is_primitive on GCC < 11, since it's a template (bug)
struct is_primitive_impl
{
  // must be a struct instead of an alias, so the access of Traits<T>::is_primitive happens in the context of this class
  template <typename T>
  struct is_primitive : ::cuda::std::bool_constant<Traits<T>::is_primitive>
  {};
};
// This trait serves two purposes:
// 1. It is used for tunings to detect whether we have a build-in arithmetic type for which we can expect certain
// arithmetic throughput. E.g.: we expect all primitive types of the same size to show roughly similar performance.
// 2. Decoupled lookback uses this trait to determine whether there is a machine word twice the size of T which can be
// loaded/stored with a single instruction.
// TODO(bgruber): for 2. we should probably just check whether sizeof(T) * 2 <= sizeof(int128) (or 256-bit on SM100)
// Users must be able to hook into both scenarios with their custom types, so this trait must depend on cub::Traits
template <typename T>
struct is_primitive : is_primitive_impl::is_primitive<T>
{};

template <typename T>
inline constexpr bool is_primitive_v = is_primitive<T>::value;
} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
