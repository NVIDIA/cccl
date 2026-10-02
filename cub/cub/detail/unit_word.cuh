// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/// Unit-words of data movement
template <typename T>
struct UnitWord
{
  CCCL_DEPRECATED static constexpr auto ALIGN_BYTES = alignof(T);

  template <typename Unit>
  struct CCCL_DEPRECATED IsMultiple
  {
    static constexpr auto UNIT_ALIGN_BYTES = alignof(Unit);
    static constexpr bool IS_MULTIPLE =
      (sizeof(T) % sizeof(Unit) == 0) && (int(alignof(T)) % int(UNIT_ALIGN_BYTES) == 0);
  };

  // MSVC < 19.39 gets an internal compile error on the __is_multiple variable template below, so use a struct instead
#  if _CCCL_COMPILER(MSVC, <, 19, 39)
  template <typename Unit>
  using __is_multiple =
    ::cuda::std::bool_constant<(sizeof(T) % sizeof(Unit) == 0) && (alignof(T) % alignof(Unit) == 0)>;

  /// Largest shuffle word such that sizeof(T) % sizeof(W) == 0 and alignof(W) <= alignof(T)
  using ShuffleWord =
    ::cuda::std::_If<__is_multiple<int>::value,
                     unsigned int,
                     ::cuda::std::_If<__is_multiple<short>::value, unsigned short, unsigned char>>;

  /// Largest volatile word such that sizeof(T) % sizeof(W) == 0 and alignof(W) <= alignof(T)
  using VolatileWord = ::cuda::std::_If<__is_multiple<long long>::value, unsigned long long, ShuffleWord>;

  /// Largest memory-access word such that sizeof(T) % sizeof(W) == 0 and alignof(W) <= alignof(T)
  using DeviceWord = ::cuda::std::_If<__is_multiple<longlong2>::value, ulonglong2, VolatileWord>;

  /// Biggest texture reference word such that sizeof(T) % sizeof(W) == 0 and alignof(W) <= alignof(T)
  using TextureWord =
    ::cuda::std::_If<__is_multiple<int4>::value, uint4, ::cuda::std::_If<__is_multiple<int2>::value, uint2, ShuffleWord>>;
#  else // _CCCL_COMPILER(MSVC, <, 19, 39)
  template <typename Unit>
  static constexpr bool __is_multiple = (sizeof(T) % sizeof(Unit) == 0) && (alignof(T) % alignof(Unit) == 0);

  /// Largest shuffle word evenly dividing T and not increasing the alignment
  using ShuffleWord = ::cuda::std::
    _If<__is_multiple<int>, unsigned int, ::cuda::std::_If<__is_multiple<short>, unsigned short, unsigned char>>;

  /// Largest volatile word evenly dividing T and not increasing the alignment
  using VolatileWord = ::cuda::std::_If<__is_multiple<long long>, unsigned long long, ShuffleWord>;

  /// Largest memory-access word evenly dividing T and not increasing the alignment
  using DeviceWord = ::cuda::std::_If<__is_multiple<longlong2>, ulonglong2, VolatileWord>;

  /// Biggest texture reference word evenly dividing T and not increasing the alignment
  using TextureWord =
    ::cuda::std::_If<__is_multiple<int4>, uint4, ::cuda::std::_If<__is_multiple<int2>, uint2, ShuffleWord>>;
#  endif // _CCCL_COMPILER(MSVC, <, 19, 39)
};

// float2 specialization workaround (for SM10-SM13)
template <>
struct UnitWord<float2>
{
  using ShuffleWord  = int;
  using VolatileWord = unsigned long long;
  using DeviceWord   = unsigned long long;
  using TextureWord  = float2;
};

// float4 specialization workaround (for SM10-SM13)
template <>
struct UnitWord<float4>
{
  using ShuffleWord  = int;
  using VolatileWord = unsigned long long;
  using DeviceWord   = ulonglong2;
  using TextureWord  = float4;
};

// char2 specialization workaround (for SM10-SM13)
template <>
struct UnitWord<char2>
{
  using ShuffleWord  = unsigned short;
  using VolatileWord = unsigned short;
  using DeviceWord   = unsigned short;
  using TextureWord  = unsigned short;
};

// clang-format off
template <typename T> struct UnitWord<volatile T> : UnitWord<T> {};
template <typename T> struct UnitWord<const T> : UnitWord<T> {};
template <typename T> struct UnitWord<const volatile T> : UnitWord<T> {};
// clang-format on
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
