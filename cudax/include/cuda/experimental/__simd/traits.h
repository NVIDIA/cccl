//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___SIMD_TRAITS_H
#define _CUDAX___SIMD_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__memory/is_sufficiently_aligned.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__simd/declaration.h>
#include <cuda/experimental/__simd/utility.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::datapar
{
struct element_aligned_tag
{
  template <class _Simd, class _Ptr>
  [[nodiscard]] _CCCL_API static constexpr _Ptr* __apply(_Ptr* __ptr) noexcept
  {
    return __ptr;
  }
};

struct vector_aligned_tag
{
  template <class _Simd, class _Ptr>
  [[nodiscard]] _CCCL_API static constexpr _Ptr* __apply(_Ptr* __ptr) noexcept
  {
    return __ptr;
  }
};

template <::cuda::std::size_t _Alignment>
struct overaligned_tag
{
  template <class _Simd, class _Ptr>
  [[nodiscard]] _CCCL_API static constexpr _Ptr* __apply(_Ptr* __ptr) noexcept
  {
    _CCCL_ASSERT(::cuda::std::is_sufficiently_aligned<_Alignment>(__ptr),
                 "Pointer does not satisfy overaligned_tag alignment requirement");
    return __ptr;
  }
};

inline constexpr element_aligned_tag element_aligned{};
inline constexpr vector_aligned_tag vector_aligned{};

template <::cuda::std::size_t _Alignment>
inline constexpr overaligned_tag<_Alignment> overaligned{};

template <typename _Tp>
inline constexpr bool is_abi_tag_v = false;

template <typename _Tp>
struct is_abi_tag : ::cuda::std::bool_constant<is_abi_tag_v<_Tp>>
{};

template <int _Np>
inline constexpr bool is_abi_tag_v<simd_abi::fixed_size<_Np>> = true;

template <typename _Tp>
inline constexpr bool is_simd_v = false;

template <typename _Tp>
struct is_simd : ::cuda::std::bool_constant<is_simd_v<_Tp>>
{};

template <typename _Tp>
inline constexpr bool is_simd_mask_v = false;

template <typename _Tp>
struct is_simd_mask : ::cuda::std::bool_constant<is_simd_mask_v<_Tp>>
{};

template <typename _Tp>
inline constexpr bool is_simd_flag_type_v = false;

template <typename _Tp>
struct is_simd_flag_type : ::cuda::std::bool_constant<is_simd_flag_type_v<_Tp>>
{};

template <typename _Tp, typename _Abi = simd_abi::fixed_size<1>, bool = (__is_vectorizable_v<_Tp> && is_abi_tag_v<_Abi>)>
struct simd_size : ::cuda::std::integral_constant<::cuda::std::size_t, _Abi::__simd_size>
{};

template <typename _Tp, typename _Abi>
struct simd_size<_Tp, _Abi, false>
{
  static constexpr ::cuda::std::size_t value = 0;
};

template <typename _Tp, typename _Abi = simd_abi::fixed_size<1>>
inline constexpr ::cuda::std::size_t simd_size_v = simd_size<_Tp, _Abi>::value;

template <typename _Tp, typename _Abi>
inline constexpr bool is_simd_v<basic_simd<_Tp, _Abi>> = true;

template <typename _Tp, typename _Abi>
inline constexpr bool is_simd_v<basic_simd_mask<_Tp, _Abi>> = true;

template <typename _Tp, typename _Abi>
inline constexpr bool is_simd_mask_v<basic_simd_mask<_Tp, _Abi>> = true;

template <>
inline constexpr bool is_simd_flag_type_v<element_aligned_tag> = true;

template <>
inline constexpr bool is_simd_flag_type_v<vector_aligned_tag> = true;

template <::cuda::std::size_t _Alignment>
inline constexpr bool is_simd_flag_type_v<overaligned_tag<_Alignment>> = true;

// Memory alignment queries
template <typename _Tp, typename _Flags = element_aligned_tag>
struct memory_alignment;

template <typename _Tp, typename _Abi>
struct memory_alignment<basic_simd<_Tp, _Abi>, element_aligned_tag>
    : ::cuda::std::integral_constant<::cuda::std::size_t, alignof(_Tp)>
{};

template <typename _Tp, typename _Abi>
struct memory_alignment<basic_simd<_Tp, _Abi>, vector_aligned_tag>
    : ::cuda::std::integral_constant<::cuda::std::size_t, alignof(_Tp) * simd_size_v<_Tp, _Abi>>
{};

template <typename _Tp, typename _Abi, ::cuda::std::size_t _Alignment>
struct memory_alignment<basic_simd<_Tp, _Abi>, overaligned_tag<_Alignment>>
    : ::cuda::std::integral_constant<::cuda::std::size_t, _Alignment>
{};

template <typename _Tp, typename _Abi>
struct memory_alignment<basic_simd_mask<_Tp, _Abi>, element_aligned_tag>
    : ::cuda::std::integral_constant<::cuda::std::size_t, alignof(bool)>
{};

template <typename _Tp, typename _Abi>
struct memory_alignment<basic_simd_mask<_Tp, _Abi>, vector_aligned_tag>
    : ::cuda::std::integral_constant<::cuda::std::size_t, alignof(bool) * simd_size_v<_Tp, _Abi>>
{};

template <typename _Tp, typename _Abi, ::cuda::std::size_t _Alignment>
struct memory_alignment<basic_simd_mask<_Tp, _Abi>, overaligned_tag<_Alignment>>
    : ::cuda::std::integral_constant<::cuda::std::size_t, _Alignment>
{};

template <typename _Tp, typename _Flags = element_aligned_tag>
inline constexpr ::cuda::std::size_t memory_alignment_v = memory_alignment<_Tp, _Flags>::value;

// Rebind simd element type
template <typename _Tp, typename _Simd>
struct rebind_simd;

template <typename _Tp, typename _Up, typename _Abi>
struct rebind_simd<_Tp, basic_simd<_Up, _Abi>>
{
  using type = basic_simd<_Tp, _Abi>;
};

template <typename _Tp, typename _Up, typename _Abi>
struct rebind_simd<_Tp, basic_simd_mask<_Up, _Abi>>
{
  using type = basic_simd_mask<_Tp, _Abi>;
};

template <typename _Tp, typename _Simd>
using rebind_simd_t = typename rebind_simd<_Tp, _Simd>::type;
} // namespace cuda::experimental::datapar

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___SIMD_TRAITS_H
