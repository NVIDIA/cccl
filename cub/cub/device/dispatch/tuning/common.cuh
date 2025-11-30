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
  other
};

template <typename T>
inline constexpr auto classify_op = op_kind_t::other;

template <typename T>
inline constexpr auto classify_op<::cuda::std::plus<T>> = op_kind_t::plus;
} // namespace detail
CUB_NAMESPACE_END
