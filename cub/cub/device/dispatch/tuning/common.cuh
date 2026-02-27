// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/is_signed.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
// copy of cccl_type_enum from cccl/c/types.h, which we cannot share, since CCCL.C's public interface does not depend on
// libcu++
enum class type_t
{
  int8,
  int16,
  int32,
  int64,
  int128,
  uint8,
  uint16,
  uint32,
  uint64,
  uint128,
  float32,
  float64,
  other
};

template <typename T>
inline constexpr auto classify_type = type_t::other;

template <>
inline constexpr auto classify_type<char> = ::cuda::std::is_signed_v<char> ? type_t::int8 : type_t::uint8;
template <>
inline constexpr auto classify_type<signed char> = type_t::int8;
template <>
inline constexpr auto classify_type<unsigned char> = type_t::uint8;

template <>
inline constexpr auto classify_type<signed short> = type_t::int16;
template <>
inline constexpr auto classify_type<unsigned short> = type_t::uint16;

template <>
inline constexpr auto classify_type<signed int> = type_t::int32;
template <>
inline constexpr auto classify_type<unsigned int> = type_t::uint32;

template <>
inline constexpr auto classify_type<signed long> = sizeof(signed long) == 4 ? type_t::int32 : type_t::int64;
template <>
inline constexpr auto classify_type<unsigned long> = sizeof(unsigned long) == 4 ? type_t::uint32 : type_t::uint64;

template <>
inline constexpr auto classify_type<signed long long> = type_t::int64;
template <>
inline constexpr auto classify_type<unsigned long long> = type_t::int64;

#if _CCCL_HAS_INT128()
template <>
inline constexpr auto classify_type<__int128_t> = type_t::int128;
template <>
inline constexpr auto classify_type<__uint128_t> = type_t::int128;
#endif // _CCCL_HAS_INT128()

template <>
inline constexpr auto classify_type<float> = type_t::float32;
template <>
inline constexpr auto classify_type<double> = type_t::float64;

// similar to cccl_op_kind_t from cccl/c/types.h
enum class op_kind_t
{
  plus,
  min,
  max,
  other
};

template <typename T>
inline constexpr auto classify_op = op_kind_t::other;

template <typename T>
inline constexpr auto classify_op<::cuda::std::plus<T>> = op_kind_t::plus;

struct iterator_info
{
  int value_type_size;
  int value_type_alignment;
  bool value_type_is_trivially_relocatable;
  bool is_contiguous;
};

template <typename It>
[[nodiscard]] _CCCL_API constexpr auto make_iterator_info() -> iterator_info
{
  using vt = it_value_t<It>;
  return iterator_info{
    static_cast<int>(size_of<vt>),
    static_cast<int>(align_of<vt>),
    THRUST_NS_QUALIFIER::is_trivially_relocatable_v<vt>,
    THRUST_NS_QUALIFIER::is_contiguous_iterator_v<It>};
}

enum class primitive_key
{
  no,
  yes
};
enum class primitive_length
{
  no,
  yes
};
enum class key_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
enum class length_size
{
  _4,
  unknown
};

template <class T>
_CCCL_API constexpr primitive_key is_primitive_key()
{
  return is_primitive<T>::value ? primitive_key::yes : primitive_key::no;
}

template <class T>
_CCCL_API constexpr primitive_length is_primitive_length()
{
  return is_primitive<T>::value ? primitive_length::yes : primitive_length::no;
}

template <class KeyT>
_CCCL_API constexpr key_size classify_key_size()
{
  return sizeof(KeyT) == 1 ? key_size::_1
       : sizeof(KeyT) == 2 ? key_size::_2
       : sizeof(KeyT) == 4 ? key_size::_4
       : sizeof(KeyT) == 8 ? key_size::_8
       : sizeof(KeyT) == 16
         ? key_size::_16
         : key_size::unknown;
}

template <class LengthT>
_CCCL_API constexpr length_size classify_length_size()
{
  return sizeof(LengthT) == 4 ? length_size::_4 : length_size::unknown;
}
} // namespace detail
CUB_NAMESPACE_END
