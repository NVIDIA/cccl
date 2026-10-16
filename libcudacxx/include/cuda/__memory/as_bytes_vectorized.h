//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_AS_BYTES_VECTORIZED_H
#define _CUDA___MEMORY_AS_BYTES_VECTORIZED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/climits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// 32-byte accesses are supported since CTK 13.0
#if _CCCL_CTK_AT_LEAST(13, 0)
inline constexpr size_t __max_vector_access = 32ull;
#else
inline constexpr size_t __max_vector_access = 16ull;
#endif // _CCCL_CTK_AT_LEAST(13, 0)

//! `_Count` unsigned integer words that are accessed together, aligned for a single vector access of their size
template <class _Word, size_t _Count>
struct alignas(sizeof(_Word) * _Count) __vector_word
{
  static_assert(sizeof(_Word) * _Count <= __max_vector_access, "A vector word must fit into a single vector access");

  _Word __storage_[_Count];
};

//! Uninitialized storage of `_NumVectors` vector words, aligned for the widest vector access that evenly divides it, at
//! most to `__max_vector_access` but at least to `_MinAlignment`
template <class _VectorWord, size_t _MinAlignment, size_t _NumVectors>
struct __bytes_vectorized_t
{
  using value_type = _VectorWord;

  // Aligned to the widest vector access that evenly divides the storage size
  static constexpr size_t __full_alignment = size_t{1} << ::cuda::std::countr_zero(sizeof(_VectorWord) * _NumVectors);
  static constexpr size_t __alignment =
    ::cuda::std::max(_MinAlignment, ::cuda::std::min(__full_alignment, __max_vector_access));

  alignas(__alignment) _VectorWord __storage_[_NumVectors];

  static_assert(sizeof(__storage_) % __alignment == 0, "Storage size must be a multiple of its alignment");

  [[nodiscard]] _CCCL_API constexpr _VectorWord* data() noexcept
  {
    return __storage_;
  }

  [[nodiscard]] _CCCL_API constexpr const _VectorWord* data() const noexcept
  {
    return __storage_;
  }

  template <class _Tp>
  [[nodiscard]] _CCCL_API _Tp* __alias() noexcept
  {
    static_assert(sizeof(_Tp) <= sizeof(__storage_), "T does not fit in vectorized storage");
    static_assert(alignof(_Tp) <= __alignment, "T is more aligned than vectorized storage");
    return ::cuda::std::launder(reinterpret_cast<_Tp*>(__storage_));
  }

  template <class _Tp>
  [[nodiscard]] _CCCL_API const _Tp* __alias() const noexcept
  {
    static_assert(sizeof(_Tp) <= sizeof(__storage_), "T does not fit in vectorized storage");
    static_assert(alignof(_Tp) <= __alignment, "T is more aligned than vectorized storage");
    return ::cuda::std::launder(reinterpret_cast<const _Tp*>(__storage_));
  }
};

template <class _Tp, size_t _VecSize>
[[nodiscard]] _CCCL_API constexpr auto __as_bytes_vectorized_impl() noexcept
{
  if constexpr (alignof(_Tp) <= __max_vector_access)
  {
    using _Word = ::cuda::std::__make_nbit_uint_t<::cuda::std::min(alignof(_Tp), size_t{8}) * CHAR_BIT>;
    constexpr size_t __num_words = sizeof(_Tp) / sizeof(_Word) * _VecSize;
    constexpr size_t __words_per_vector =
      ::cuda::std::min(size_t{1} << ::cuda::std::countr_zero(__num_words), __max_vector_access / sizeof(_Word));
    return __bytes_vectorized_t<__vector_word<_Word, __words_per_vector>,
                                alignof(_Tp),
                                __num_words / __words_per_vector>{};
  }
  else
  { // Default case: use 32 bit words as the vectorized storage
    return __bytes_vectorized_t<::cuda::std::uint32_t, alignof(_Tp), ::cuda::ceil_div(sizeof(_Tp), 4) * _VecSize>{};
  }
}

//! Storage for `_VecSize` objects of type `_Tp`, made of unsigned integer words of the alignment of `_Tp`, at most 8
//! bytes, grouped into the widest vector words that evenly divide the storage. It is aligned for the widest vector
//! access of its size, at most to `__max_vector_access` but at least to `alignof(_Tp)`
template <class _Tp, size_t _VecSize>
using __as_bytes_vectorized_t = decltype(::cuda::__as_bytes_vectorized_impl<_Tp, _VecSize>());

template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL bool __is_vectorizable() noexcept
{
  // More aligned types would need a vectorized storage aligned beyond the widest vector access
  return alignof(_Tp) <= __max_vector_access;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_AS_BYTES_VECTORIZED_H
