// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * \file
 * Wrappers and extensions around <type_traits> utilities.
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

#include <cub/util_cpp_dialect.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/__concepts/concept_macros.h> // IWYU pragma: keep
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/mdspan>
#include <cuda/std/span>
#include <cuda/std/type_traits> // is_same_v

CUB_NAMESPACE_BEGIN
namespace detail
{

template <typename T, typename... TArgs>
inline constexpr bool is_one_of_v = (_CUDA_VSTD::is_same_v<T, TArgs> || ...);

template <typename T, typename V, typename = void>
struct has_binary_call_operator : _CUDA_VSTD::false_type
{};

template <typename T, typename V>
struct has_binary_call_operator<
  T,
  V,
  _CUDA_VSTD::void_t<decltype(_CUDA_VSTD::declval<T>()(_CUDA_VSTD::declval<V>(), _CUDA_VSTD::declval<V>()))>>
    : _CUDA_VSTD::true_type
{};

/***********************************************************************************************************************
 * Array-like type traits
 **********************************************************************************************************************/

template <typename T>
inline constexpr bool is_fixed_size_random_access_range_v = false;

template <typename T, size_t N>
inline constexpr bool is_fixed_size_random_access_range_v<T[N]> = true;

template <typename T, size_t N>
inline constexpr bool is_fixed_size_random_access_range_v<_CUDA_VSTD::array<T, N>> = true;

template <typename T, size_t N>
inline constexpr bool is_fixed_size_random_access_range_v<_CUDA_VSTD::span<T, N>> = N != _CUDA_VSTD::dynamic_extent;

template <typename T, typename E, typename L, typename A>
inline constexpr bool is_fixed_size_random_access_range_v<_CUDA_VSTD::mdspan<T, E, L, A>> =
  E::rank == 1 && E::rank_dynamic() == 0;

/***********************************************************************************************************************
 * static_size: a type trait that returns the number of elements in an Array-like type
 **********************************************************************************************************************/

template <typename T>
inline constexpr int static_size_v = _CUDA_VSTD::enable_if_t<_CUDA_VSTD::__always_false_v<T>>{};

template <typename T, size_t N>
inline constexpr int static_size_v<T[N]> = N;

template <typename T, size_t N>
inline constexpr int static_size_v<_CUDA_VSTD::array<T, N>> = N;

template <typename T, size_t N>
inline constexpr int static_size_v<_CUDA_VSTD::span<T, N>> =
  _CUDA_VSTD::enable_if_t<N != _CUDA_VSTD::dynamic_extent, int>{N};

template <typename T, typename E, typename L, typename A>
inline constexpr int static_size_v<_CUDA_VSTD::mdspan<T, E, L, A>> =
  _CUDA_VSTD::enable_if_t<E::rank == 1 && E::rank_dynamic() == 0, int>{E::static_extent(0)};

template <typename T>
using implicit_prom_t = decltype(+T{});

/***********************************************************************************************************************
 * Extended floating point traits
 **********************************************************************************************************************/
// half

template <typename>
inline constexpr bool is_half_impl_v = false;

template <typename>
inline constexpr bool is_half2_impl_v = false;

#if _CCCL_HAS_NVFP16()

template <>
inline constexpr bool is_half_impl_v<__half> = true;

template <>
inline constexpr bool is_half2_impl_v<__half2> = true;

#endif // _CCCL_HAS_NVFP16

template <typename T>
inline constexpr bool is_half_v = is_half_impl_v<_CUDA_VSTD::remove_cv_t<T>>;

template <typename T>
inline constexpr bool is_half2_v = is_half2_impl_v<_CUDA_VSTD::remove_cv_t<T>>;

template <typename T>
inline constexpr bool is_any_half_v = is_half_impl_v<T> || is_half2_impl_v<T>;

//----------------------------------------------------------------------------------------------------------------------
// bfloat16

template <typename>
inline constexpr bool is_bfloat16_impl_v = false;

template <typename>
inline constexpr bool is_bfloat162_impl_v = false;

#if _CCCL_HAS_NVBF16()

template <>
inline constexpr bool is_bfloat16_impl_v<__nv_bfloat16> = true;

template <>
inline constexpr bool is_bfloat162_impl_v<__nv_bfloat162> = true;

#endif // _CCCL_HAS_NVBF16

template <typename T>
inline constexpr bool is_bfloat16_v = is_bfloat16_impl_v<_CUDA_VSTD::remove_cv_t<T>>;

template <typename T>
inline constexpr bool is_bfloat162_v = is_bfloat162_impl_v<_CUDA_VSTD::remove_cv_t<T>>;

template <typename T>
inline constexpr bool is_any_bfloat16_v = is_bfloat16_v<T> || is_bfloat162_v<T>;

//----------------------------------------------------------------------------------------------------------------------
// short2/ushort2

template <typename T>
inline constexpr bool is_any_short2_impl_v = false;

template <>
inline constexpr bool is_any_short2_impl_v<short2> = true;

template <>
inline constexpr bool is_any_short2_impl_v<ushort2> = true;

template <typename T>
inline constexpr bool is_any_short2_v = is_any_short2_impl_v<_CUDA_VSTD::remove_cv_t<T>>;

//----------------------------------------------------------------------------------------------------------------------

// - promote small integer types to their corresponding 32-bit promotion type
// - address the incompatibility between linux/windows for int/long
template <typename T>
using signed_promotion_t = _CUDA_VSTD::_If<
  _CUDA_VSTD::__cccl_is_signed_integer_v<T> && sizeof(T) <= sizeof(int),
  int,
  _CUDA_VSTD::_If<_CUDA_VSTD::__cccl_is_unsigned_integer_v<T> && sizeof(T) <= sizeof(uint32_t), uint32_t, T>>;

} // namespace detail
CUB_NAMESPACE_END
