//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_ITERATOR_H
#define _CUDA_STD___SIMD_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/default_sentinel.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_const.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.iterator], class template __simd_iterator
template <typename _Vp>
class __simd_iterator
{
  _Vp* __data_               = nullptr;
  __simd_size_type __offset_ = 0;

  _CCCL_API constexpr __simd_iterator(_Vp& __data, const __simd_size_type __offset) noexcept
      : __data_{::cuda::std::addressof(__data)}
      , __offset_{__offset}
  {
    _CCCL_ASSERT(__data_ != nullptr, "cuda::std::simd::__simd_iterator: data is nullptr");
    _CCCL_ASSERT(::cuda::in_range(__offset_, __simd_size_type{0}, _Vp::__size),
                 "cuda::std::simd::__simd_iterator: offset is out of range");
  }

  template <typename, typename, typename>
  friend class basic_vec;

  template <size_t, typename, typename>
  friend class basic_mask;

  template <typename>
  friend class __simd_iterator;

public:
  using value_type        = typename _Vp::value_type;
  using iterator_category = input_iterator_tag;
  using iterator_concept  = random_access_iterator_tag;
  using difference_type   = __simd_size_type;

  _CCCL_HIDE_FROM_ABI constexpr __simd_iterator() noexcept                                  = default;
  _CCCL_HIDE_FROM_ABI constexpr __simd_iterator(const __simd_iterator&) noexcept            = default;
  _CCCL_HIDE_FROM_ABI constexpr __simd_iterator& operator=(const __simd_iterator&) noexcept = default;

  // non-const to const converting constructor
  // workaround for MSVC (cannot used is_const_v<_Vp>)
  // _Vp = const T:  const T == const T
  // _Vp = T:        const T != T
  _CCCL_TEMPLATE(typename _Up = remove_const_t<_Vp>)
  _CCCL_REQUIRES(is_same_v<const _Up, _Vp>)
  _CCCL_API constexpr __simd_iterator(const __simd_iterator<_Up>& __i) noexcept
      : __data_{__i.__data_}
      , __offset_{__i.__offset_}
  {}

  [[nodiscard]] _CCCL_API constexpr value_type operator*() const noexcept
  {
    _CCCL_ASSERT(__data_ != nullptr, "cuda::std::simd::__simd_iterator: data is nullptr");
    return (*__data_)[__offset_];
  }

  _CCCL_API constexpr __simd_iterator& operator++() noexcept
  {
    ++__offset_;
    _CCCL_ASSERT(::cuda::in_range(__offset_, __simd_size_type{0}, _Vp::__size),
                 "cuda::std::simd::__simd_iterator: offset is out of range");
    return *this;
  }

  _CCCL_API constexpr __simd_iterator operator++(int) noexcept
  {
    const __simd_iterator __tmp = *this;
    ++__offset_;
    _CCCL_ASSERT(::cuda::in_range(__offset_, __simd_size_type{0}, _Vp::__size),
                 "cuda::std::simd::__simd_iterator: offset is out of range");
    return __tmp;
  }

  _CCCL_API constexpr __simd_iterator& operator--() noexcept
  {
    --__offset_;
    _CCCL_ASSERT(::cuda::in_range(__offset_, __simd_size_type{0}, _Vp::__size),
                 "cuda::std::simd::__simd_iterator: offset is out of range");
    return *this;
  }

  _CCCL_API constexpr __simd_iterator operator--(int) noexcept
  {
    const __simd_iterator __tmp = *this;
    --__offset_;
    _CCCL_ASSERT(::cuda::in_range(__offset_, __simd_size_type{0}, _Vp::__size),
                 "cuda::std::simd::__simd_iterator: offset is out of range");
    return __tmp;
  }

  _CCCL_API constexpr __simd_iterator& operator+=(const difference_type __n) noexcept
  {
    __offset_ += __n;
    _CCCL_ASSERT(::cuda::in_range(__offset_, __simd_size_type{0}, _Vp::__size),
                 "cuda::std::simd::__simd_iterator: offset is out of range");
    return *this;
  }

  _CCCL_API constexpr __simd_iterator& operator-=(const difference_type __n) noexcept
  {
    __offset_ -= __n;
    _CCCL_ASSERT(::cuda::in_range(__offset_, __simd_size_type{0}, _Vp::__size),
                 "cuda::std::simd::__simd_iterator: offset is out of range");
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr value_type operator[](const difference_type __n) const noexcept
  {
    _CCCL_ASSERT(__data_ != nullptr, "cuda::std::simd::__simd_iterator: data is nullptr");
    _CCCL_ASSERT(::cuda::in_range(__offset_ + __n, __simd_size_type{0}, _Vp::__size - 1),
                 "cuda::std::simd::__simd_iterator: offset is out of range");
    return (*__data_)[__offset_ + __n];
  }

  // [simd.iterator] comparisons

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __simd_iterator __a, const __simd_iterator __b) noexcept
  {
    return __a.__data_ == __b.__data_ && __a.__offset_ == __b.__offset_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __simd_iterator __i, default_sentinel_t) noexcept
  {
    return __i.__offset_ == _Vp::__size;
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator!=(const __simd_iterator __a, const __simd_iterator __b) noexcept
  {
    return !(__a == __b);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const __simd_iterator __i, const default_sentinel_t __s) noexcept
  {
    return !(__i == __s);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const default_sentinel_t __s, const __simd_iterator __i) noexcept
  {
    return !(__i == __s);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const default_sentinel_t __s, const __simd_iterator __i) noexcept
  {
    return __i == __s;
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]]
  _CCCL_API friend constexpr auto operator<=>(const __simd_iterator __a, const __simd_iterator __b) noexcept
  {
    _CCCL_ASSERT(__a.__data_ == __b.__data_, "cuda::std::simd::__simd_iterator: iterators refer to different objects");
    return __a.__offset_ <=> __b.__offset_;
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  [[nodiscard]]
  _CCCL_API friend constexpr bool operator<(const __simd_iterator __a, const __simd_iterator __b) noexcept
  {
    _CCCL_ASSERT(__a.__data_ == __b.__data_, "cuda::std::simd::__simd_iterator: iterators refer to different objects");
    return __a.__offset_ < __b.__offset_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>(const __simd_iterator __a, const __simd_iterator __b) noexcept
  {
    _CCCL_ASSERT(__a.__data_ == __b.__data_, "cuda::std::simd::__simd_iterator: iterators refer to different objects");
    return __b.__offset_ < __a.__offset_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const __simd_iterator __a, const __simd_iterator __b) noexcept
  {
    return !(__b < __a);
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const __simd_iterator __a, const __simd_iterator __b) noexcept
  {
    return !(__a < __b);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  // [simd.iterator] arithmetic

  [[nodiscard]] _CCCL_API friend constexpr __simd_iterator
  operator+(__simd_iterator __i, const difference_type __n) noexcept
  {
    return __i += __n;
  }

  [[nodiscard]] _CCCL_API friend constexpr __simd_iterator
  operator+(const difference_type __n, __simd_iterator __i) noexcept
  {
    return __i += __n;
  }

  [[nodiscard]] _CCCL_API friend constexpr __simd_iterator
  operator-(__simd_iterator __i, const difference_type __n) noexcept
  {
    return __i -= __n;
  }

  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const __simd_iterator __a, const __simd_iterator __b) noexcept
  {
    _CCCL_ASSERT(__a.__data_ == __b.__data_, "cuda::std::simd::__simd_iterator: iterators refer to different objects");
    return __a.__offset_ - __b.__offset_;
  }

  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const __simd_iterator __i, default_sentinel_t) noexcept
  {
    return __i.__offset_ - _Vp::__size;
  }

  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(default_sentinel_t, const __simd_iterator __i) noexcept
  {
    return _Vp::__size - __i.__offset_;
  }
};

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <typename _Vp>
struct iterator_traits<simd::__simd_iterator<_Vp>>
{
  using __iter            = simd::__simd_iterator<_Vp>;
  using iterator_concept  = typename __iter::iterator_concept;
  using iterator_category = typename __iter::iterator_category;
  using value_type        = typename __iter::value_type;
  using difference_type   = typename __iter::difference_type;
  using pointer           = void;
  using reference         = value_type;
};

_CCCL_END_NAMESPACE_CUDA_STD

#if _CCCL_HAS_HOST_STD_LIB()
_CCCL_BEGIN_NAMESPACE_STD

template <typename _Diff, typename _Vp>
_CCCL_HOST_API constexpr void advance(::cuda::std::simd::__simd_iterator<_Vp>& __iter, const _Diff __diff) noexcept
{
  ::cuda::std::advance(__iter, __diff);
}

template <typename _Vp>
[[nodiscard]] _CCCL_HOST_API constexpr typename ::cuda::std::simd::__simd_iterator<_Vp>::difference_type distance(
  const ::cuda::std::simd::__simd_iterator<_Vp> __first, const ::cuda::std::simd::__simd_iterator<_Vp> __last) noexcept
{
  return ::cuda::std::distance(__first, __last);
}

template <typename _Vp>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::simd::__simd_iterator<_Vp>
next(::cuda::std::simd::__simd_iterator<_Vp> __iter,
     const typename ::cuda::std::simd::__simd_iterator<_Vp>::difference_type __n = 1) noexcept
{
  ::cuda::std::advance(__iter, __n);
  return __iter;
}

template <typename _Vp>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::simd::__simd_iterator<_Vp>
prev(::cuda::std::simd::__simd_iterator<_Vp> __iter,
     const typename ::cuda::std::simd::__simd_iterator<_Vp>::difference_type __n = 1) noexcept
{
  ::cuda::std::advance(__iter, -__n);
  return __iter;
}

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HAS_HOST_STD_LIB()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_ITERATOR_H
