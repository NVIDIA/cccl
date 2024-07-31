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
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__utility/select_execution_space.h>

#if _CCCL_STD_VER >= 2014 && !defined(_CCCL_COMPILER_MSVC_2017) \
  && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

//! @file The \c heterogeneous_iterator class provides a iterator that provides typed execution space safety.
namespace cuda::experimental
{

template <class _Tp, bool _IsConst, _ExecutionSpace _Space>
class heterogeneous_iterator;

// We restrict all accessors of the iterator based on the execution space
template <class _Tp, bool _IsConst, _ExecutionSpace _Space>
class __heterogeneous_iterator_access;

template <class _Tp, bool _IsConst>
class __heterogeneous_iterator_access<_Tp, _IsConst, _ExecutionSpace::__host>
{
public:
#  if _CCCL_STD_VER >= 2014
  using iterator_concept = _CUDA_VSTD::contiguous_iterator_tag;
#  endif // _CCCL_STD_VER >= 2014
  using iterator_category = _CUDA_VSTD::random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = _CUDA_VSTD::ptrdiff_t;
  using pointer           = _CUDA_VSTD::__maybe_const<_IsConst, _Tp>*;
  using reference         = _CUDA_VSTD::__maybe_const<_IsConst, _Tp>&;

  __heterogeneous_iterator_access() = default;

  _CCCL_HOST_DEVICE explicit constexpr __heterogeneous_iterator_access(pointer __ptr) noexcept
      : __ptr_(__ptr)
  {}

  //! @brief Dereference a heterogeneous_iterator
  //! @return A reference to the element the iterator points to
  _CCCL_NODISCARD _CCCL_HOST constexpr reference operator*() const noexcept
  {
    return *__ptr_;
  }

  //! @brief Operator arrow on a heterogeneous_iterator
  //! @return A pointer to the element the iterator points to
  _CCCL_NODISCARD _CCCL_HOST constexpr pointer operator->() const noexcept
  {
    return __ptr_;
  }

  //! @brief Dereference a heterogeneous_iterator
  //! @param __count The offset at which we want to dereference
  //! @return A reference of the \p __count th element after the one the iterator points to
  _CCCL_NODISCARD _CCCL_HOST constexpr reference operator[](const difference_type __count) const noexcept
  {
    return *(__ptr_ + __count);
  }

private:
  pointer __ptr_ = nullptr;

  template <class, bool, _ExecutionSpace>
  friend class heterogeneous_iterator;

  template <class>
  friend struct pointer_traits;
};

template <class _Tp, bool _IsConst>
class __heterogeneous_iterator_access<_Tp, _IsConst, _ExecutionSpace::__device>
{
public:
#  if _CCCL_STD_VER >= 2014
  using iterator_concept = _CUDA_VSTD::contiguous_iterator_tag;
#  endif // _CCCL_STD_VER >= 2014
  using iterator_category = _CUDA_VSTD::random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = _CUDA_VSTD::ptrdiff_t;
  using pointer           = _CUDA_VSTD::__maybe_const<_IsConst, _Tp>*;
  using reference         = _CUDA_VSTD::__maybe_const<_IsConst, _Tp>&;

  __heterogeneous_iterator_access() = default;

  _CCCL_HOST_DEVICE explicit constexpr __heterogeneous_iterator_access(pointer __ptr) noexcept
      : __ptr_(__ptr)
  {}

  //! @brief Dereference a heterogeneous_iterator
  //! @return A reference to the element the iterator points to
  _CCCL_NODISCARD _CCCL_DEVICE constexpr reference operator*() const noexcept
  {
    return *__ptr_;
  }

  //! @brief Operator arrow on a heterogeneous_iterator
  //! @return A pointer to the element the iterator points to
  _CCCL_NODISCARD _CCCL_DEVICE constexpr pointer operator->() const noexcept
  {
    return __ptr_;
  }

  //! @brief Dereference a heterogeneous_iterator
  //! @param __count The offset at which we want to dereference
  //! @return A reference of the \p __count th element after the one the iterator points to
  _CCCL_NODISCARD _CCCL_DEVICE constexpr reference operator[](const difference_type __count) const noexcept
  {
    return *(__ptr_ + __count);
  }

private:
  pointer __ptr_ = nullptr;

  template <class, bool, _ExecutionSpace>
  friend class heterogeneous_iterator;

  template <class>
  friend struct pointer_traits;
};

template <class _Tp, bool _IsConst>
class __heterogeneous_iterator_access<_Tp, _IsConst, _ExecutionSpace::__host_device>
{
public:
#  if _CCCL_STD_VER >= 2014
  using iterator_concept = _CUDA_VSTD::contiguous_iterator_tag;
#  endif // _CCCL_STD_VER >= 2014
  using iterator_category = _CUDA_VSTD::random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = _CUDA_VSTD::ptrdiff_t;
  using pointer           = _CUDA_VSTD::__maybe_const<_IsConst, _Tp>*;
  using reference         = _CUDA_VSTD::__maybe_const<_IsConst, _Tp>&;

  __heterogeneous_iterator_access() = default;

  _CCCL_HOST_DEVICE explicit constexpr __heterogeneous_iterator_access(pointer __ptr) noexcept
      : __ptr_(__ptr)
  {}

  //! @brief Dereference a heterogeneous_iterator
  //! @return A reference to the element the iterator points to
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reference operator*() const noexcept
  {
    return *__ptr_;
  }

  //! @brief Operator arrow on a heterogeneous_iterator
  //! @return A pointer to the element the iterator points to
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr pointer operator->() const noexcept
  {
    return __ptr_;
  }

  //! @brief Dereference a heterogeneous_iterator
  //! @param __count The offset at which we want to dereference
  //! @return A reference to the `__count` element after the one the iterator points to
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reference operator[](const difference_type __count) const noexcept
  {
    return *(__ptr_ + __count);
  }

private:
  pointer __ptr_ = nullptr;

  template <class, bool, _ExecutionSpace>
  friend class heterogeneous_iterator;

  template <class>
  friend struct pointer_traits;
};

template <class _Tp, bool _IsConst, _ExecutionSpace _Space>
class heterogeneous_iterator : public __heterogeneous_iterator_access<_Tp, _IsConst, _Space>
{
  template <class>
  friend struct pointer_traits;

public:
#  if _CCCL_STD_VER >= 2014
  using iterator_concept = _CUDA_VSTD::contiguous_iterator_tag;
#  endif // _CCCL_STD_VER >= 2014
  using iterator_category = _CUDA_VSTD::random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = _CUDA_VSTD::ptrdiff_t;
  using pointer           = _CUDA_VSTD::__maybe_const<_IsConst, _Tp>*;
  using reference         = _CUDA_VSTD::__maybe_const<_IsConst, _Tp>&;

  heterogeneous_iterator() = default;

  //! @brief Construct a heterogeneous_iterator from a pointer to the underlying memory
  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator(pointer __ptr) noexcept
      : __heterogeneous_iterator_access<_Tp, _IsConst, _Space>(__ptr)
  {}

  //! @brief Converting constructor from a mutable iterator to a non-mutable one
  _LIBCUDACXX_TEMPLATE(bool _OtherConst)
  _LIBCUDACXX_REQUIRES((_OtherConst != _IsConst) _LIBCUDACXX_AND _IsConst)
  _CCCL_HOST_DEVICE explicit constexpr heterogeneous_iterator(
    heterogeneous_iterator<_Tp, _OtherConst, _Space> __other) noexcept
      : __heterogeneous_iterator_access<_Tp, _IsConst, _Space>(__other.__ptr_)
  {}

  //! @brief Increment of a heterogeneous_iterator
  //! @return The heterogeneous_iterator pointing to the next element
  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator& operator++() noexcept
  {
    ++this->__ptr_;
    return *this;
  }

  //! @brief Post-increment of a heterogeneous_iterator
  //! @return A copy of the heterogeneous_iterator pointing to the next element
  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator operator++(int) noexcept
  {
    heterogeneous_iterator __temp = *this;
    ++this->__ptr_;
    return __temp;
  }

  //! @brief Decrement of a heterogeneous_iterator
  //! @return The heterogeneous_iterator pointing to the previous element
  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator& operator--() noexcept
  {
    --this->__ptr_;
    return *this;
  }

  //! @brief Post-decrement of a heterogeneous_iterator
  //! @return A copy of the heterogeneous_iterator pointing to the previous element
  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator operator--(int) noexcept
  {
    heterogeneous_iterator __temp = *this;
    --this->__ptr_;
    return __temp;
  }

  //! @brief Advance a heterogeneous_iterator
  //! @param __count The number of elements to advance
  //! @return The heterogeneous_iterator advanced by \p __count
  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator& operator+=(const difference_type __count) noexcept
  {
    this->__ptr_ += __count;
    return *this;
  }

  //! @brief Advance a heterogeneous_iterator
  //! @param __count The number of elements to advance
  //! @return A copy of this heterogeneous_iterator advanced by \p __count
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr heterogeneous_iterator operator+(const difference_type __count) noexcept
  {
    heterogeneous_iterator __temp = *this;
    __temp += __count;
    return __temp;
  }

  //! @brief Advance a heterogeneous_iterator
  //! @param __count The number of elements to advance
  //! @param __other A heterogeneous_iterator
  //! @return \p __other advanced by \p __count
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr heterogeneous_iterator
  operator+(const difference_type __count, heterogeneous_iterator __other) noexcept
  {
    __other += __count;
    return __other;
  }

  //! @brief Advance a heterogeneous_iterator by the negative value of \p __count
  //! @param __count The number of elements to advance
  //! @return The heterogeneous_iterator advanced by the negative value of \p __count
  _CCCL_HOST_DEVICE constexpr heterogeneous_iterator& operator-=(const difference_type __count) noexcept
  {
    this->__ptr_ -= __count;
    return *this;
  }

  //! @brief Advance a heterogeneous_iterator by the negative value of \p __count
  //! @param __count The number of elements to advance
  //! @return A copy of this heterogeneous_iterator advanced by the negative value of \p __count
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr heterogeneous_iterator operator-(const difference_type __count) noexcept
  {
    heterogeneous_iterator __temp = *this;
    __temp -= __count;
    return __temp;
  }

  //! @brief Distance between two heterogeneous_iterator
  //! @param __other The other heterogeneous_iterator
  //! @return The distance between the two elements the heterogeneous_iterator point to
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr difference_type operator-(const heterogeneous_iterator& __other) noexcept
  {
    return static_cast<difference_type>(this->__ptr_ - __other.__ptr_);
  }

  //! @brief Equality comparison between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator
  //! @param __rhs Another heterogeneous_iterator
  //! @return true, if both heterogeneous_iterator point to the same element
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator==(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ == __rhs.__ptr_;
  }
#  if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator
  //! @param __rhs Another heterogeneous_iterator
  //! @return false, if both heterogeneous_iterator point to the same element
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
  //! @brief Less then relation between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator
  //! @param __rhs Another heterogeneous_iterator
  //! @return true, if the address of the element pointed to by \p __lhs is less then the address of the one pointed to
  //! by \p __rhs
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator<(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ < __rhs.__ptr_;
  }
  //! @brief Less equal relation between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator
  //! @param __rhs Another heterogeneous_iterator
  //! @return true, if the address of the element pointed to by \p __lhs is less then or equal to the address of the one
  //! pointed to by \p __rhs
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator<=(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ <= __rhs.__ptr_;
  }
  //! @brief Greater then relation between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator
  //! @param __rhs Another heterogeneous_iterator
  //! @return true, if the address of the element pointed to by \p __lhs is greater then the address of the one
  //! pointed to by \p __rhs
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator>(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ > __rhs.__ptr_;
  }
  //! @brief Greater equal relation between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator
  //! @param __rhs Another heterogeneous_iterator
  //! @return true, if the address of the element pointed to by \p __lhs is greater then or equal to the address of the
  //! one pointed to by \p __rhs
  _CCCL_NODISCARD_FRIEND _CCCL_HOST_DEVICE constexpr bool
  operator>=(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ >= __rhs.__ptr_;
  }
#  endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
};
} // namespace cuda::experimental

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Here be dragons: We need to ensure that the iterator can work with legacy interfaces that take a pointer.
// This will obviously eat all of our execution checks
template <class _Tp, bool _IsConst, ::cuda::experimental::_ExecutionSpace _Space>
struct pointer_traits<::cuda::experimental::heterogeneous_iterator<_Tp, _IsConst, _Space>>
{
  using pointer         = ::cuda::experimental::heterogeneous_iterator<_Tp, _IsConst, _Space>;
  using element_type    = __maybe_const<_IsConst, typename pointer::value_type>;
  using difference_type = _CUDA_VSTD::ptrdiff_t;

  //! @brief Retrieve the address of the element pointed at by an heterogeneous_iterator
  //! @param __iter A heterogeneous_iterator
  //! @return A pointer to the element pointed to by the heterogeneous_iterator
  _CCCL_NODISCARD _CCCL_HOST_DEVICE static constexpr element_type* to_address(const pointer __iter) noexcept
  {
    return _CUDA_VSTD::to_address(__iter.__ptr_);
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER >= 2014 && !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //__CUDAX__CONTAINERS_HETEROGENEOUS_ITERATOR_H
