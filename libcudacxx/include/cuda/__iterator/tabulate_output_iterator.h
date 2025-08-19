//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_TABULATE_OUTPUT_ITERATOR_H
#define _CUDA___ITERATOR_TABULATE_OUTPUT_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Fn, class _Index = ::cuda::std::ptrdiff_t>
class tabulate_output_iterator;

template <class _Fn, class _Index>
class __tabulate_proxy
{
private:
  template <class, class>
  friend class tabulate_output_iterator;

  _Fn& __func_;
  _Index __index_;

public:
  _CCCL_API constexpr explicit __tabulate_proxy(_Fn& __func, _Index __index) noexcept
      : __func_(__func)
      , __index_(__index)
  {}

  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES(::cuda::std::is_invocable_v<_Fn&, _Index, _Arg> _CCCL_AND(
    !::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Arg>, __tabulate_proxy>))
  _CCCL_API constexpr const __tabulate_proxy&
  operator=(_Arg&& __arg) noexcept(::cuda::std::is_nothrow_invocable_v<_Fn&, _Index, _Arg>)
  {
    ::cuda::std::invoke(__func_, __index_, ::cuda::std::forward<_Arg>(__arg));
    return *this;
  }

  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES(::cuda::std::is_invocable_v<const _Fn&, _Index, _Arg> _CCCL_AND(
    !::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Arg>, __tabulate_proxy>))
  _CCCL_API constexpr const __tabulate_proxy& operator=(_Arg&& __arg) const
    noexcept(::cuda::std::is_nothrow_invocable_v<const _Fn&, _Index, _Arg>)
  {
    ::cuda::std::invoke(__func_, __index_, ::cuda::std::forward<_Arg>(__arg));
    return *this;
  }
};

//! @p tabulate_output_iterator is a special kind of output iterator which, whenever a value is assigned to a
//! dereferenced iterator, calls the given callable with the index that corresponds to the offset of the dereferenced
//! iterator and the assigned value.
//!
//! The following code snippet demonstrated how to create a \p tabulate_output_iterator which prints the index and the
//! assigned value.
//!
//! @code
//! #include <cuda/iterator>
//!
//! struct print_op
//! {
//!   __host__ __device__ void operator()(int index, float value) const
//!   {
//!     printf("%d: %f\n", index, value);
//!   }
//! };
//!
//! int main()
//! {
//!   auto tabulate_it = cuda::make_tabulate_output_iterator(print_op{});
//!
//!   tabulate_it[0] =  1.0f;    // prints: 0: 1.0
//!   tabulate_it[1] =  3.0f;    // prints: 1: 3.0
//!   tabulate_it[9] =  5.0f;    // prints: 9: 5.0
//! }
//! @endcode
template <class _Fn, class _Index>
class tabulate_output_iterator
{
private:
  ::cuda::std::ranges::__movable_box<_Fn> __func_;
  _Index __index_ = 0;

public:
  using iterator_concept  = ::cuda::std::random_access_iterator_tag;
  using iterator_category = ::cuda::std::random_access_iterator_tag;
  using difference_type   = _Index;
  using value_type        = void;
  using pointer           = void;
  using reference         = void;

#if _CCCL_HAS_CONCEPTS()
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI tabulate_output_iterator()
    requires ::cuda::std::default_initializable<_Fn>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Fn2 = _Fn)
  _CCCL_REQUIRES(::cuda::std::default_initializable<_Fn2>)
  _CCCL_API constexpr tabulate_output_iterator() noexcept(::cuda::std::is_nothrow_default_constructible_v<_Fn2>) {}
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  //! @brief Constructs a \p tabulate_output_iterator with a given functor \p __func and and optional index
  _CCCL_API constexpr tabulate_output_iterator(_Fn __func, _Index __index = 0) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Fn>)
      : __func_(::cuda::std::in_place, ::cuda::std::move(__func))
      , __index_(__index)
  {}

  //! @brief Returns the stored index
  [[nodiscard]] _CCCL_API constexpr difference_type index() const noexcept
  {
    return __index_;
  }

  //! @brief Dereferences the \c tabulate_output_iterator returning a proxy that applies the stored function and index
  //! on assignment
  [[nodiscard]] _CCCL_API constexpr auto operator*() const noexcept
  {
    return __tabulate_proxy<_Fn, _Index>{const_cast<_Fn&>(*__func_), __index_};
  }

  //! @brief Dereferences the \c tabulate_output_iterator returning a proxy that applies the stored function and index
  //! on assignment
  [[nodiscard]] _CCCL_API constexpr auto operator*() noexcept
  {
    return __tabulate_proxy<_Fn, _Index>{*__func_, __index_};
  }

  //! @brief Subscripts the \c tabulate_output_iterator returning a proxy that applies the stored function and advanced
  //! index on assignment
  //! @param __n The additional offset to advance the stored index
  [[nodiscard]] _CCCL_API constexpr auto operator[](difference_type __n) const noexcept
  {
    return __tabulate_proxy<_Fn, _Index>{const_cast<_Fn&>(*__func_), __index_ + __n};
  }

  //! @brief Subscripts the \c tabulate_output_iterator returning a proxy that applies the stored function and advanced
  //! index on assignment
  //! @param __n The additional offset to advance the stored index
  [[nodiscard]] _CCCL_API constexpr auto operator[](difference_type __n) noexcept
  {
    return __tabulate_proxy<_Fn, _Index>{*__func_, __index_ + __n};
  }

  //! @brief Increments the stored index
  _CCCL_API constexpr tabulate_output_iterator& operator++() noexcept
  {
    ++__index_;
    return *this;
  }

  //! @brief Increments the stored index
  _CCCL_API constexpr tabulate_output_iterator
  operator++(int) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
  {
    tabulate_output_iterator __tmp = *this;
    ++__index_;
    return __tmp;
  }

  //! @brief Decrements the stored index
  _CCCL_API constexpr tabulate_output_iterator& operator--() noexcept
  {
    --__index_;
    return *this;
  }

  //! @brief Decrements the stored index
  _CCCL_API constexpr tabulate_output_iterator
  operator--(int) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
  {
    tabulate_output_iterator __tmp = *this;
    --__index_;
    return __tmp;
  }

  //! @brief Returns a copy of this \c tabulate_output_iterator advanced by \p __n
  //! @param __n The number of elements to advance
  [[nodiscard]] _CCCL_API constexpr tabulate_output_iterator operator+(difference_type __n) const
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
  {
    return tabulate_output_iterator{*__func_, __index_ + __n};
  }

  //! @brief Returns a copy of a \c tabulate_output_iterator \p __iter advanced by \p __n
  //! @param __n The number of elements to advance
  //! @param __iter The original \c tabulate_output_iterator
  [[nodiscard]] _CCCL_API friend constexpr tabulate_output_iterator
  operator+(difference_type __n,
            const tabulate_output_iterator& __iter) noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
  {
    return __iter + __n;
  }

  //! @brief Advances the index of this \c tabulate_output_iterator by \p __n
  //! @param __n The number of elements to advance
  _CCCL_API constexpr tabulate_output_iterator& operator+=(difference_type __n) noexcept
  {
    __index_ += __n;
    return *this;
  }

  //! @brief Returns a copy of this \c tabulate_output_iterator decremented by \p __n
  //! @param __n The number of elements to decrement
  [[nodiscard]] _CCCL_API constexpr tabulate_output_iterator operator-(difference_type __n) const
    noexcept(::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
  {
    return tabulate_output_iterator{*__func_, __index_ - __n};
  }

  //! @brief Returns the distance between two \c tabulate_output_iterator 's
  [[nodiscard]] _CCCL_API friend constexpr difference_type
  operator-(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __rhs.__index_ - __lhs.__index_;
  }

  //! @brief Decrements the index of the \c tabulate_output_iterator by \p __n
  //! @param __n The number of elements to decrement
  _CCCL_API constexpr tabulate_output_iterator& operator-=(difference_type __n) noexcept
  {
    __index_ -= __n;
    return *this;
  }

  //! @brief Compares two \c tabulate_output_iterator for equality by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index_ == __rhs.__index_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two \c tabulate_output_iterator for inequality by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index_ != __rhs.__index_;
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way-compares two \c tabulate_output_iterator by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr strong_ordering
  operator<=>(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index_ <=> __rhs.__index_;
  }
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  //! @brief Compares two \c tabulate_output_iterator for less than by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index_ < __rhs.__index_;
  }

  //! @brief Compares two \c tabulate_output_iterator for less equal by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index_ <= __rhs.__index_;
  }

  //! @brief Compares two \c tabulate_output_iterator for greater than by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index_ > __rhs.__index_;
  }

  //! @brief Compares two \c tabulate_output_iterator for greater equal by comparing their indices
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const tabulate_output_iterator& __lhs, const tabulate_output_iterator& __rhs) noexcept
  {
    return __lhs.__index_ >= __rhs.__index_;
  }
};

template <class _Fn>
_CCCL_HOST_DEVICE tabulate_output_iterator(_Fn) -> tabulate_output_iterator<_Fn, ::cuda::std::ptrdiff_t>;

_CCCL_TEMPLATE(class _Fn, class _Index)
_CCCL_REQUIRES(::cuda::std::__integer_like<_Index>)
_CCCL_HOST_DEVICE tabulate_output_iterator(_Fn, _Index) -> tabulate_output_iterator<_Fn, _Index>;

//! @brief Creates a \p tabulate_output_iterator from an optional index.
//! @param __index The index of the \p tabulate_output_iterator within a range. The default index is \c 0.
//! @return A new \p tabulate_output_iterator with \p __index as the couner.
_CCCL_TEMPLATE(class _Fn, class _Integer = ::cuda::std::ptrdiff_t)
_CCCL_REQUIRES(::cuda::std::__integer_like<_Integer>)
[[nodiscard]] _CCCL_API constexpr auto make_tabulate_output_iterator(_Fn __func, _Integer __index = 0)
{
  return tabulate_output_iterator{::cuda::std::move(__func), __index};
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_TABULATE_OUTPUT_ITERATOR_H
