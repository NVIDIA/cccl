//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINERS_HETEROGENEOUS_ITERATOR_H
#define __CUDAX__CONTAINERS_HETEROGENEOUS_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/properties.h>
#include <cuda/std/__concepts/_One_of.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/cstdint>

#if _CCCL_STD_VER >= 2014 && !defined(_CCCL_COMPILER_MSVC_2017) \
  && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

//! @file The \c heterogeneous_iterator class provides a iterator that provides typed execution space safety.
namespace cuda::experimental
{

template <class... _Properties>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_host_accessible =
  _CUDA_VSTD::_One_of<_CUDA_VMR::host_accessible, _OtherProperties...>;

template <class... _Properties>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_device_accessible =
  _CUDA_VSTD::_One_of<_CUDA_VMR::device_accessible, _OtherProperties...>;

template <class... _Properties>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_host_device_accessible =
  __is_host_accessible<_Properties...> && __is_device_accessible<_Properties...>;

enum class _ExecutionSpace
{
  __host,
  __device,
  __host_device,
};

template <class... _Properties>
_LIBCUDACXX_INLINE_VAR constexpr _ExecutionSpace __select_execution_space =
  __is_host_device_accessible<_Properties...> ? _ExecutionSpace::__host_device
  : __is_device_accessible<_Properties...>
    ? _ExecutionSpace::__device
    : _ExecutionSpace::__host;

template <class _Tp, bool _IsConst, _ExecutionSpace _Space>
class heterogeneous_iterator;

// We restrict all accessors of the iterator based on the execution space
template <class _Tp, bool _IsConst, _ExecutionSpace _Space>
class __heterogeneous_iterator_access;

template <class _Tp, bool _IsConst>
class __heterogeneous_iterator_access<_Tp, _IsConst, _ExecutionSpace::__host>
{
private:
  _Tp* __ptr_ = nullptr;

  friend class heterogeneous_iterator;

  template <class>
  friend struct pointer_traits;

public:
#  if _CCCL_STD_VER >= 2014
  using iterator_concept = contiguous_iterator_tag;
#  endif // _CCCL_STD_VER >= 2014
  using iterator_category = random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = _CUDA_VSTD::ptrdiff_t;
  using pointer           = __maybe_const<_Tp*, _IsConst>;
  using reference         = __maybe_const<_Tp&, _IsConst>;

  __heterogeneous_iterator_access() = default;

  _CCCL_HOST_DEVICE constexpr __heterogeneous_iterator_access(_Tp* __ptr) noexcept
      : __ptr_(__ptr)
  {}

  _CCCL_NODISCARD _CCCL_HOST constexpr reference operator*() const noexcept
  {
    return *__ptr_;
  }

  _CCCL_NODISCARD _CCCL_HOST constexpr pointer operator->() const noexcept
  {
    return __ptr_;
  }

  _CCCL_NODISCARD _CCCL_HOST constexpr reference operator[](const difference_type __count) const noexcept
  {
    return *(__ptr_ + __count);
  }
};

template <class _Tp, bool _IsConst>
class __heterogeneous_iterator_access<_Tp, _IsConst, _ExecutionSpace::__device>
{
private:
  _Tp* __ptr_ = nullptr;

  friend class heterogeneous_iterator;

  template <class>
  friend struct pointer_traits;

public:
#  if _CCCL_STD_VER >= 2014
  using iterator_concept = contiguous_iterator_tag;
#  endif // _CCCL_STD_VER >= 2014
  using iterator_category = random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = _CUDA_VSTD::ptrdiff_t;
  using pointer           = __maybe_const<_Tp*, _IsConst>;
  using reference         = __maybe_const<_Tp&, _IsConst>;

  __heterogeneous_iterator_access() = default;

  _CCCL_HOST_DEVICE constexpr __heterogeneous_iterator_access(_Tp* __ptr) noexcept
      : __ptr_(__ptr)
  {}

  _CCCL_NODISCARD _CCCL_DEVICE constexpr reference operator*() const noexcept
  {
    return *__ptr_;
  }

  _CCCL_NODISCARD _CCCL_DEVICE constexpr pointer operator->() const noexcept
  {
    return __ptr_;
  }

  _CCCL_NODISCARD _CCCL_DEVICE constexpr reference operator[](const difference_type __count) const noexcept
  {
    return *(__ptr_ + __count);
  }
};

template <class _Tp, bool _IsConst>
class __heterogeneous_iterator_access<_Tp, _IsConst, _ExecutionSpace::__host_device>
{
private:
  _Tp* __ptr_ = nullptr;

  friend class heterogeneous_iterator;

  template <class>
  friend struct pointer_traits;

public:
#  if _CCCL_STD_VER >= 2014
  using iterator_concept = contiguous_iterator_tag;
#  endif // _CCCL_STD_VER >= 2014
  using iterator_category = random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = _CUDA_VSTD::ptrdiff_t;
  using pointer           = __maybe_const<_Tp*, _IsConst>;
  using reference         = __maybe_const<_Tp&, _IsConst>;

  __heterogeneous_iterator_access() = default;

  _CCCL_HOST_DEVICE constexpr __heterogeneous_iterator_access(_Tp* __ptr) noexcept
      : __ptr_(__ptr)
  {}

  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reference operator*() const noexcept
  {
    return *__ptr_;
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr pointer operator->() const noexcept
  {
    return __ptr_;
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reference operator[](const difference_type __count) const noexcept
  {
    return *(__ptr_ + __count);
  }
};

template <class _Tp, bool _IsConst, _ExecutionSpace _Space>
class heterogeneous_iterator : public __heterogeneous_iterator_access<_Tp, _IsConst, _Space>
{
  template <class>
  friend struct pointer_traits;

public:
  heterogeneous_iterator() = default;

  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator(_Tp* __ptr) noexcept
      : __heterogeneous_iterator_access<_Tp, _IsConst, _Space>(__ptr)
  {}

  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator& operator++() noexcept
  {
    ++this->__ptr_;
    return *this;
  }
  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator operator++(int) noexcept
  {
    heterogeneous_iterator __temp = *this;
    ++this->__ptr_;
    return __temp;
  }

  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator& operator--() noexcept
  {
    --this->__ptr_;
    return *this;
  }
  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator operator--(int) noexcept
  {
    heterogeneous_iterator __temp = *this;
    --this->__ptr_;
    return __temp;
  }

  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator& operator+=(const difference_type __count) noexcept
  {
    this->__ptr_ += __count;
    return *this;
  }
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr heterogeneous_iterator operator+(const difference_type __count) noexcept
  {
    heterogeneous_iterator __temp = *this;
    __temp += __count;
    return __temp;
  }
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr heterogeneous_iterator
  operator+(const difference_type __count, heterogeneous_iterator __other) noexcept
  {
    __other += __count;
    return __other;
  }

  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator& operator-=(const difference_type __count) noexcept
  {
    this->__ptr_ -= __count;
    return *this;
  }
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr heterogeneous_iterator operator-(const difference_type __count) noexcept
  {
    heterogeneous_iterator __temp = *this;
    __temp -= __count;
    return __temp;
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr difference_type operator-(const heterogeneous_iterator& __other) noexcept
  {
    return static_cast<difference_type>(this->__ptr_ - __other.__ptr_);
  }

  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator==(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ == __rhs.__ptr_;
  }
#  if _CCCL_STD_VER <= 2017
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator!=(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ != __rhs.__ptr_;
  }
#  endif // _CCCL_STD_VER <= 2017

#  ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr _CUDA_VSTD::strong_ordering
  operator<=>(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ <=> __rhs.__ptr_;
  }
#  else // ^^^ _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR ^^^ /  vvv !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR vvv
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator<(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ < __rhs.__ptr_;
  }
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator<=(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ <= __rhs.__ptr_;
  }
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator>(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ > __rhs.__ptr_;
  }
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator>=(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ >= __rhs.__ptr_;
  }
#  endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
};

// Here be dragons: We need to ensure that the iterator can work with legacy interfaces that take a pointer.
// This will obviously eat all of our execution checks
template <class _Tp, bool _IsConst, _ExecutionSpace _Space>
struct pointer_traits<heterogeneous_iterator<_Tp, _IsConst, _Space>>
{
  using pointer         = heterogeneous_iterator<_Tp, _IsConst, _Space>;
  using element_type    = __maybe_const<typename pointer::value_type, _IsConst>;
  using difference_type = _CUDA_VSTD::ptrdiff_t;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE static constexpr element_type* to_address(const pointer __iter) noexcept
  {
    return _CUDA_VSTD::to_address(__iter.__ptr_);
  }
};

} // namespace cuda::experimental

#endif // _CCCL_STD_VER >= 2014 && !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //__CUDAX__CONTAINERS_HETEROGENEOUS_ITERATOR_H
