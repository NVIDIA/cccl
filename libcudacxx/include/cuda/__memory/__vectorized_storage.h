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

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/climits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// 32-byte accesses are supported since PTX ISA 9.0, and only for global memory on sm_100 and newer
inline constexpr size_t __max_memory_access_width = (__cccl_ptx_isa >= 900ULL) ? 32ull : 16ull;

//! `_NumWords` words of `_WordType`, aligned to `_Alignment` bytes
//!
//! The compiler copies the storage with accesses of up to `_Alignment` bytes. The alignment must divide the size, so
//! that an array of storage has the layout of the objects it holds. Storage of another alignment is assigned word by
//! word, so that the same words can be loaded with accesses of different widths.
template <class _WordType, size_t _NumWords, size_t _Alignment>
struct __vectorized_storage_t
{
  static_assert(_NumWords > 0, "Vectorized storage must hold at least one word");
  static_assert(alignof(_WordType) <= _Alignment, "Vectorized storage must be at least as aligned as its words");
  static_assert(sizeof(_WordType) * _NumWords % _Alignment == 0, "The alignment must divide the size of the storage");

  alignas(_Alignment) _WordType __storage_[_NumWords];

  template <size_t _OtherAlignment>
  _CCCL_API constexpr __vectorized_storage_t&
  operator=(const __vectorized_storage_t<_WordType, _NumWords, _OtherAlignment>& __other) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (size_t __i = 0; __i < _NumWords; ++__i)
    {
      __storage_[__i] = __other.__storage_[__i];
    }
    return *this;
  }

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
    static_assert(sizeof(__storage_) % sizeof(_Tp) == 0, "T does not fit in vectorized storage");
    static_assert(alignof(_Tp) <= _Alignment, "T is more aligned than vectorized storage");
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
    static_assert(sizeof(__storage_) % sizeof(_Tp) == 0, "T does not fit in vectorized storage");
    static_assert(alignof(_Tp) <= _Alignment, "T is more aligned than vectorized storage");
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

//! The width in bytes of the words in which `__as_vectorized_storage_t` holds `_Count` objects of `_Tp`
//!
//! - Words of at least 4 bytes keep objects narrower than a register packed in registers, so that the compiler does
//!   not split a vector into its objects before the code that uses them.
//! - Words as wide as the alignment of the objects, up to 8 bytes, keep the padding of padded objects within words of
//!   their members, so that the compiler does not drop words of padding and split a vector load around them.
//! - The words are no wider than the largest power of two that divides the storage, so that they divide it.
template <class _Tp, size_t _Count>
[[nodiscard]] _CCCL_API constexpr size_t __vectorized_storage_word_size() noexcept
{
  static_assert(_Count > 0, "Vectorized storage must hold at least one object");
  const size_t __size_alignment = size_t{1} << ::cuda::std::countr_zero(sizeof(_Tp) * _Count);
  return ::cuda::std::min(::cuda::std::clamp(alignof(_Tp), size_t{4}, size_t{8}), __size_alignment);
}

//! The words in which `__as_vectorized_storage_t` holds `_Count` objects of `_Tp`
template <class _Tp, size_t _Count>
using __vectorized_storage_word_t _CCCL_NODEBUG =
  ::cuda::std::__make_nbit_uint_t<::cuda::__vectorized_storage_word_size<_Tp, _Count>() * CHAR_BIT>;

//! The alignment of the widest access to `_Count` objects of `_Tp`: the largest power of two that divides their size,
//! at most `__max_memory_access_width`, but at least the alignment of `_Tp`
template <class _Tp, size_t _Count>
inline constexpr size_t __vectorized_storage_alignment_v = ::cuda::std::max(
  alignof(_Tp),
  ::cuda::std::min(size_t{1} << ::cuda::std::countr_zero(sizeof(_Tp) * _Count), __max_memory_access_width));

//! Storage for `_Tp[_Count]` as words of `__vectorized_storage_word_t<_Tp, _Count>`, aligned to `_Alignment` bytes,
//! but at least to the alignment of `_Tp`
template <class _Tp, size_t _Count, size_t _Alignment = __vectorized_storage_alignment_v<_Tp, _Count>>
using __as_vectorized_storage_t _CCCL_NODEBUG =
  __vectorized_storage_t<__vectorized_storage_word_t<_Tp, _Count>,
                         sizeof(_Tp) * _Count / sizeof(__vectorized_storage_word_t<_Tp, _Count>),
                         ::cuda::std::max(alignof(_Tp), _Alignment)>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY___VECTORIZED_STORAGE_H
