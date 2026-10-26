//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY___VECTORIZED_STORAGE_H
#define _CUDA___MEMORY___VECTORIZED_STORAGE_H

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
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/climits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// 32-byte accesses are supported since PTX ISA 9.0, and only for global memory on sm_100 and newer
inline constexpr size_t __max_vector_access = (__cccl_ptx_isa >= 900ULL) ? 32ull : 16ull;

//! Uninitialized storage of `_NumVectors` vectors of `_VectorSize` words, aligned for the widest vector access that
//! evenly divides it, at most to `_MaxVectorAccess` but at least to `_MinAlignment`
//!
//! - The alignment determines how the storage is loaded. The compiler splits a copy of the storage into the widest
//!   loads that its alignment and size allow, so storage that is only aligned to its words is loaded word by word. The
//!   alignment must divide the size, so that the storage has no padding and an array of storage has the layout of the
//!   objects it holds.
//! - `_WordType` determines how the objects are extracted after the load. Extracting an object from a wider word takes
//!   a single shift or byte permutation, but an object that spans several narrower words is reassembled from them
//!   piece by piece. Words as wide as the alignment of the objects avoid the reassembly.
//! - 32-bit words suit any type aligned to more than `_MaxVectorAccess`, whatever the layout of its members. They have
//!   the width of a register, so members aligned to 4 bytes or more occupy whole registers, while wider words can make
//!   the compiler split the copy into narrower loads or rebuild the members from bytes.
template <class _WordType, size_t _NumVectors, size_t _VectorSize, size_t _MinAlignment, size_t _MaxVectorAccess>
struct __vectorized_storage_t
{
  static_assert(sizeof(_WordType) * _VectorSize <= _MaxVectorAccess, "A vector must fit into a single vector access");

  using value_type = _WordType[_VectorSize];

  // Aligned to the widest vector access that evenly divides the storage size
  static constexpr size_t __full_alignment = size_t{1} << ::cuda::std::countr_zero(sizeof(value_type) * _NumVectors);
  static constexpr size_t __alignment =
    ::cuda::std::max(_MinAlignment, ::cuda::std::min(__full_alignment, _MaxVectorAccess));

  // A single array, so that a pointer to the first word can be iterated over the whole storage
  alignas(__alignment) _WordType __storage_[_NumVectors * _VectorSize];

  static_assert(sizeof(__storage_) % __alignment == 0, "Storage size must be a multiple of its alignment");

  [[nodiscard]] _CCCL_API constexpr _WordType* data() noexcept
  {
    return __storage_;
  }

  [[nodiscard]] _CCCL_API constexpr const _WordType* data() const noexcept
  {
    return __storage_;
  }

  //! Returns the `__index`-th object of type `_Tp` in the storage
  template <class _Tp, class _Index>
  [[nodiscard]] _CCCL_API _Tp& __alias(const _Index __index) noexcept
  {
    static_assert(::cuda::std::__cccl_is_integer_v<_Index>, "The index must be an integer");
    static_assert(sizeof(_Tp) <= sizeof(__storage_), "T does not fit in vectorized storage");
    static_assert(alignof(_Tp) <= __alignment, "T is more aligned than vectorized storage");
    if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Tp>, _WordType>)
    {
      return __storage_[__index];
    }
    else
    {
      return ::cuda::std::launder(reinterpret_cast<_Tp*>(__storage_))[__index];
    }
  }

  //! Returns the `__index`-th object of type `_Tp` in the storage
  template <class _Tp, class _Index>
  [[nodiscard]] _CCCL_API const _Tp& __alias(const _Index __index) const noexcept
  {
    static_assert(::cuda::std::__cccl_is_integer_v<_Index>, "The index must be an integer");
    static_assert(sizeof(_Tp) <= sizeof(__storage_), "T does not fit in vectorized storage");
    static_assert(alignof(_Tp) <= __alignment, "T is more aligned than vectorized storage");
    if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Tp>, _WordType>)
    {
      return __storage_[__index];
    }
    else
    {
      return ::cuda::std::launder(reinterpret_cast<const _Tp*>(__storage_))[__index];
    }
  }
};

template <class _Tp, size_t _VecSize, size_t _MaxVectorAccess>
[[nodiscard]] _CCCL_API constexpr auto __as_vectorized_storage_impl() noexcept
{
  if constexpr (alignof(_Tp) <= _MaxVectorAccess)
  {
    using _Word _CCCL_NODEBUG =
      // Arithmetic types use themselves as the word, except for 128-bit integers. The compiler handles those as two
      // 64-bit integers, which breaks the vectorized load.
      ::cuda::std::conditional_t<::cuda::std::is_arithmetic_v<_Tp> && sizeof(_Tp) <= 8ull,
                                 ::cuda::std::remove_cv_t<_Tp>,
                                 ::cuda::std::__make_nbit_uint_t<::cuda::std::min(alignof(_Tp), size_t{8}) * CHAR_BIT>>;
    constexpr size_t __num_words = sizeof(_Tp) / sizeof(_Word) * _VecSize;
    constexpr size_t __words_per_vector =
      ::cuda::std::min(size_t{1} << ::cuda::std::countr_zero(__num_words), _MaxVectorAccess / sizeof(_Word));
    return __vectorized_storage_t<_Word,
                                  __num_words / __words_per_vector,
                                  __words_per_vector,
                                  alignof(_Tp),
                                  _MaxVectorAccess>{};
  }
  else
  { // Default case: use 32 bit words as the vectorized storage
    return __vectorized_storage_t<::cuda::std::uint32_t,
                                  ::cuda::ceil_div(sizeof(_Tp), sizeof(::cuda::std::uint32_t)) * _VecSize,
                                  1,
                                  alignof(_Tp),
                                  _MaxVectorAccess>{};
  }
}

//! Storage for `_VecSize` objects of type `_Tp`. Arithmetic types of at most 8 bytes use themselves as the word. Other
//! types use unsigned integer words of the alignment of `_Tp`, at most 8 bytes. Words are grouped into the widest
//! vectors that evenly divide the storage. Types aligned to more than `_MaxVectorAccess` use vectors of a single 32-bit
//! word instead. The storage is aligned for the widest vector access of its size, at most to `_MaxVectorAccess` but at
//! least to `alignof(_Tp)`
template <class _Tp, size_t _VecSize, size_t _MaxVectorAccess = __max_vector_access>
using __as_vectorized_storage_t _CCCL_NODEBUG =
  decltype(::cuda::__as_vectorized_storage_impl<_Tp, _VecSize, _MaxVectorAccess>());

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY___VECTORIZED_STORAGE_H
