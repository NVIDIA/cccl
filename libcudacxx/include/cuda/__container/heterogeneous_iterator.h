//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINERS_HETEROGENEOUS_ITERATOR_CUH
#define __CUDAX__CONTAINERS_HETEROGENEOUS_ITERATOR_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__memory_resource/properties.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__memory/pointer_traits.h>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/maybe_const.h>
#  include <cuda/std/__type_traits/remove_const.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

//! @file The \c heterogeneous_iterator class is an iterator that provides typed execution space safety.
_CCCL_BEGIN_NAMESPACE_CUDA

enum class _IsConstIter
{
  __no,
  __yes,
};
//! @rst
//! .. _cudax-containers-heterogeneous-iterator:
//!
//! Type safe iterator over heterogeneous memory
//! ---------------------------------------------
//!
//! ``heterogeneous_iterator`` provides a type safe access over heterogeneous memory. Depending on whether the memory is
//! tagged as host-accessible and / or device-accessible the iterator restricts memory access.
//! All operations that do not require memory access are always available on host and device.
//!
//! @endrst
//! @tparam _Tp The underlying type of the elements the \c heterogeneous_iterator points at.
//! @tparam _IsConst Enumeration choosing whether the \c heterogeneous_iterator allows mutating the element pointed to.
//! @tparam _Properties The properties that the \c heterogeneous_iterator is tagged with.
template <class _CvTp, class... _Properties>
class heterogeneous_iterator;

// We restrict all accessors of the iterator based on the execution space
template <class _Tp, _IsConstIter _IsConst, ::cuda::mr::__memory_accessability _Space>
class __heterogeneous_iterator_access;

template <class _Tp, _IsConstIter _IsConst>
class __heterogeneous_iterator_access<_Tp, _IsConst, ::cuda::mr::__memory_accessability::__host>
{
public:
  using iterator_concept  = ::cuda::std::contiguous_iterator_tag;
  using iterator_category = ::cuda::std::random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = ::cuda::std::ptrdiff_t;
  using pointer           = ::cuda::std::__maybe_const<_IsConst == _IsConstIter::__yes, _Tp>*;
  using reference         = ::cuda::std::__maybe_const<_IsConst == _IsConstIter::__yes, _Tp>&;

  _CCCL_HIDE_FROM_ABI __heterogeneous_iterator_access() = default;

  _CCCL_API explicit constexpr __heterogeneous_iterator_access(pointer __ptr) noexcept
      : __ptr_(__ptr)
  {}

  //! @brief Dereference a \c heterogeneous_iterator
  //! @return A reference to the element the iterator points to
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST constexpr reference operator*() const noexcept
  {
    return *__ptr_;
  }

  //! @brief Operator arrow on a \c heterogeneous_iterator
  //! @return A pointer to the element the iterator points to
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST constexpr pointer operator->() const noexcept
  {
    return __ptr_;
  }

  //! @brief Dereference a \c heterogeneous_iterator
  //! @param __count The offset at which we want to dereference
  //! @return A reference of the \p __count th element after the one the iterator points to
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST constexpr reference
  operator[](const difference_type __count) const noexcept
  {
    return *(__ptr_ + __count);
  }

protected:
  pointer __ptr_ = nullptr;

  template <class, class...>
  friend class heterogeneous_iterator;
};

template <class _Tp, _IsConstIter _IsConst>
class __heterogeneous_iterator_access<_Tp, _IsConst, ::cuda::mr::__memory_accessability::__device>
{
public:
  using iterator_concept  = ::cuda::std::contiguous_iterator_tag;
  using iterator_category = ::cuda::std::random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = ::cuda::std::ptrdiff_t;
  using pointer           = ::cuda::std::__maybe_const<_IsConst == _IsConstIter::__yes, _Tp>*;
  using reference         = ::cuda::std::__maybe_const<_IsConst == _IsConstIter::__yes, _Tp>&;

  _CCCL_HIDE_FROM_ABI __heterogeneous_iterator_access() = default;

  _CCCL_API explicit constexpr __heterogeneous_iterator_access(pointer __ptr) noexcept
      : __ptr_(__ptr)
  {}

  //! @brief Dereference a \c heterogeneous_iterator
  //! @return A reference to the element the iterator points to
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr reference operator*() const noexcept
  {
    return *__ptr_;
  }

  //! @brief Operator arrow on a \c heterogeneous_iterator
  //! @return A pointer to the element the iterator points to
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr pointer operator->() const noexcept
  {
    return __ptr_;
  }

  //! @brief Dereference a \c heterogeneous_iterator
  //! @param __count The offset at which we want to dereference
  //! @return A reference of the \p __count th element after the one the iterator points to
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr reference
  operator[](const difference_type __count) const noexcept
  {
    return *(__ptr_ + __count);
  }

protected:
  pointer __ptr_ = nullptr;

  template <class, class...>
  friend class heterogeneous_iterator;
};

template <class _Tp, _IsConstIter _IsConst>
class __heterogeneous_iterator_access<_Tp, _IsConst, ::cuda::mr::__memory_accessability::__host_device>
{
public:
  using iterator_concept  = ::cuda::std::contiguous_iterator_tag;
  using iterator_category = ::cuda::std::random_access_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = ::cuda::std::ptrdiff_t;
  using pointer           = ::cuda::std::__maybe_const<_IsConst == _IsConstIter::__yes, _Tp>*;
  using reference         = ::cuda::std::__maybe_const<_IsConst == _IsConstIter::__yes, _Tp>&;

  _CCCL_HIDE_FROM_ABI __heterogeneous_iterator_access() = default;

  _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE explicit constexpr __heterogeneous_iterator_access(pointer __ptr) noexcept
      : __ptr_(__ptr)
  {}

  //! @brief Dereference a \c heterogeneous_iterator
  //! @return A reference to the element the iterator points to
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr reference operator*() const noexcept
  {
    return *__ptr_;
  }

  //! @brief Operator arrow on a \c heterogeneous_iterator
  //! @return A pointer to the element the iterator points to
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr pointer operator->() const noexcept
  {
    return __ptr_;
  }

  //! @brief Dereference a \c heterogeneous_iterator
  //! @param __count The offset at which we want to dereference
  //! @return A reference to the `__count` element after the one the iterator points to
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE constexpr reference
  operator[](const difference_type __count) const noexcept
  {
    return *(__ptr_ + __count);
  }

protected:
  pointer __ptr_ = nullptr;

  template <class, class...>
  friend class heterogeneous_iterator;
};

template <class _CvTp, class... _Properties>
class heterogeneous_iterator
    : public __heterogeneous_iterator_access<::cuda::std::remove_const_t<_CvTp>,
                                             ::cuda::std::is_const_v<_CvTp> ? _IsConstIter::__yes : _IsConstIter::__no,
                                             ::cuda::mr::__memory_accessability_from_properties<_Properties...>::value>
{
  using __base =
    __heterogeneous_iterator_access<::cuda::std::remove_const_t<_CvTp>,
                                    ::cuda::std::is_const_v<_CvTp> ? _IsConstIter::__yes : _IsConstIter::__no,
                                    ::cuda::mr::__memory_accessability_from_properties<_Properties...>::value>;

public:
  using iterator_concept  = ::cuda::std::contiguous_iterator_tag;
  using iterator_category = ::cuda::std::random_access_iterator_tag;
  using value_type        = ::cuda::std::remove_const_t<_CvTp>;
  using difference_type   = ::cuda::std::ptrdiff_t;
  using pointer           = _CvTp*;
  using reference         = _CvTp&;

  _CCCL_HIDE_FROM_ABI heterogeneous_iterator() = default;

  //! @brief Construct a \c heterogeneous_iterator from a pointer to the underlying memory
  _CCCL_API constexpr heterogeneous_iterator(pointer __ptr) noexcept
      : __base(__ptr)
  {}

  //! @brief Constructs an immutable \c heterogeneous_iterator from a mutable one
  //! @param __other The mutable \c heterogeneous_iterator
  _CCCL_TEMPLATE(class _OtherTp, class _CvTp2 = _CvTp)
  _CCCL_REQUIRES((::cuda::std::is_same_v<_OtherTp, value_type>) _CCCL_AND(::cuda::std::is_const_v<_CvTp2>))
  _CCCL_API constexpr heterogeneous_iterator(heterogeneous_iterator<_OtherTp, _Properties...> __other) noexcept
      : __base(__other.__ptr_)
  {}

  //! @brief Increment of a \c heterogeneous_iterator
  //! @return The heterogeneous_iterator pointing to the next element
  _CCCL_API constexpr heterogeneous_iterator& operator++() noexcept
  {
    ++this->__ptr_;
    return *this;
  }

  //! @brief Post-increment of a \c heterogeneous_iterator
  //! @return A copy of the heterogeneous_iterator pointing to the next element
  _CCCL_API constexpr heterogeneous_iterator operator++(int) noexcept
  {
    heterogeneous_iterator __temp = *this;
    ++this->__ptr_;
    return __temp;
  }

  //! @brief Decrement of a \c heterogeneous_iterator
  //! @return The heterogeneous_iterator pointing to the previous element
  _CCCL_API constexpr heterogeneous_iterator& operator--() noexcept
  {
    --this->__ptr_;
    return *this;
  }

  //! @brief Post-decrement of a \c heterogeneous_iterator
  //! @return A copy of the heterogeneous_iterator pointing to the previous element
  _CCCL_API constexpr heterogeneous_iterator operator--(int) noexcept
  {
    heterogeneous_iterator __temp = *this;
    --this->__ptr_;
    return __temp;
  }

  //! @brief Advance a \c heterogeneous_iterator
  //! @param __count The number of elements to advance.
  //! @return The heterogeneous_iterator advanced by \p __count
  _CCCL_API constexpr heterogeneous_iterator& operator+=(const difference_type __count) noexcept
  {
    this->__ptr_ += __count;
    return *this;
  }

  //! @brief Advance a \c heterogeneous_iterator
  //! @param __count The number of elements to advance.
  //! @return A copy of this heterogeneous_iterator advanced by \p __count
  [[nodiscard]] _CCCL_API constexpr heterogeneous_iterator operator+(const difference_type __count) const noexcept
  {
    heterogeneous_iterator __temp = *this;
    __temp += __count;
    return __temp;
  }

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  //! @brief Advance a \c heterogeneous_iterator
  //! @param __count The number of elements to advance.
  //! @param __other A heterogeneous_iterator.
  //! @return \p __other advanced by \p __count
  [[nodiscard]] _CCCL_API friend constexpr heterogeneous_iterator
  operator+(const difference_type __count, heterogeneous_iterator __other) noexcept
  {
    __other += __count;
    return __other;
  }
#  endif // _CCCL_DOXYGEN_INVOKED

  //! @brief Advance a \c heterogeneous_iterator by the negative value of \p __count
  //! @param __count The number of elements to advance.
  //! @return The heterogeneous_iterator advanced by the negative value of \p __count
  _CCCL_API constexpr heterogeneous_iterator& operator-=(const difference_type __count) noexcept
  {
    this->__ptr_ -= __count;
    return *this;
  }

  //! @brief Advance a \c heterogeneous_iterator by the negative value of \p __count
  //! @param __count The number of elements to advance.
  //! @return A copy of this heterogeneous_iterator advanced by the negative value of \p __count
  [[nodiscard]] _CCCL_API constexpr heterogeneous_iterator operator-(const difference_type __count) const noexcept
  {
    heterogeneous_iterator __temp = *this;
    __temp -= __count;
    return __temp;
  }

  //! @brief Distance between two heterogeneous_iterator
  //! @param __other The other heterogeneous_iterator.
  //! @return The distance between the two elements the heterogeneous_iterator point to
  [[nodiscard]] _CCCL_API constexpr difference_type operator-(const heterogeneous_iterator& __other) const noexcept
  {
    return static_cast<difference_type>(this->__ptr_ - __other.__ptr_);
  }

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  //! @brief Equality comparison between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator.
  //! @param __rhs Another heterogeneous_iterator.
  //! @return true, if both heterogeneous_iterator point to the same element
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ == __rhs.__ptr_;
  }
#    if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator.
  //! @param __rhs Another heterogeneous_iterator.
  //! @return false, if both heterogeneous_iterator point to the same element
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ != __rhs.__ptr_;
  }
#    endif // _CCCL_STD_VER <= 2017

#    if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  [[nodiscard]] _CCCL_API friend constexpr ::cuda::std::strong_ordering
  operator<=>(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ <=> __rhs.__ptr_;
  }
#    else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ /  vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  //! @brief Less than relation between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator.
  //! @param __rhs Another heterogeneous_iterator.
  //! @return true, if the address of the element pointed to by \p __lhs is less then the address of the one pointed to
  //! by \p __rhs
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ < __rhs.__ptr_;
  }
  //! @brief Less equal relation between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator.
  //! @param __rhs Another heterogeneous_iterator.
  //! @return true, if the address of the element pointed to by \p __lhs is less then or equal to the address of the one
  //! pointed to by \p __rhs
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ <= __rhs.__ptr_;
  }
  //! @brief Greater then relation between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator.
  //! @param __rhs Another heterogeneous_iterator.
  //! @return true, if the address of the element pointed to by \p __lhs is greater then the address of the one
  //! pointed to by \p __rhs
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ > __rhs.__ptr_;
  }
  //! @brief Greater equal relation between two heterogeneous_iterator
  //! @param __lhs A heterogeneous_iterator.
  //! @param __rhs Another heterogeneous_iterator.
  //! @return true, if the address of the element pointed to by \p __lhs is greater then or equal to the address of the
  //! one pointed to by \p __rhs
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const heterogeneous_iterator& __lhs, const heterogeneous_iterator& __rhs) noexcept
  {
    return __lhs.__ptr_ >= __rhs.__ptr_;
  }
#    endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  endif // _CCCL_DOXYGEN_INVOKED

  _CCCL_API constexpr pointer __unwrap() const noexcept
  {
    return this->__ptr_;
  }
};
_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Here be dragons: We need to ensure that the iterator can work with legacy interfaces that take a pointer.
// This will obviously eat all of our execution checks
template <class _Tp, class... _Properties>
struct pointer_traits<::cuda::heterogeneous_iterator<_Tp, _Properties...>>
{
  using pointer         = ::cuda::heterogeneous_iterator<_Tp, _Properties...>;
  using element_type    = _Tp;
  using difference_type = ::cuda::std::ptrdiff_t;

  //! @brief Retrieve the address of the element pointed at by an heterogeneous_iterator
  //! @param __iter A heterogeneous_iterator.
  //! @return A pointer to the element pointed to by the heterogeneous_iterator
  [[nodiscard]] _CCCL_API static constexpr element_type* to_address(const pointer __iter) noexcept
  {
    return ::cuda::std::to_address(__iter.__unwrap());
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //__CUDAX__CONTAINERS_HETEROGENEOUS_ITERATOR_CUH
