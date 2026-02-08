// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___ITERATOR_ZIP_ITERATOR_H
#define _CUDA___ITERATOR_ZIP_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/zip_iterator.h>
#include <cuda/std/__algorithm/ranges_min_element.h>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/three_way_comparable.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/__iterator/zip_common.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/tuple>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @addtogroup iterators
//! @{

struct __zv_iter_category_base_none
{};

struct __zv_iter_category_base_tag
{
  using iterator_category = ::cuda::std::input_iterator_tag;
};

template <class... _Iterators>
using __zv_iter_category_base =
  ::cuda::std::conditional_t<__zip_iter_constraints<_Iterators...>::__all_forward,
                             __zv_iter_category_base_tag,
                             __zv_iter_category_base_none>;

//! @addtogroup iterators
//! @{

//! @brief @c zip_iterator is an iterator which represents a @c tuple of iterators. This iterator is useful for creating
//! a virtual array of structures while achieving the same performance and bandwidth as the structure of arrays idiom.
//! @c zip_iterator also facilitates kernel fusion by providing a convenient means of amortizing the execution of the
//! same operation over multiple ranges.
//!
//! The following code snippet demonstrates how to create a @c zip_iterator which represents the result of "zipping"
//! multiple ranges together.
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//!
//! thrust::device_vector<int> int_v{0, 1, 2};
//! thrust::device_vector<float> float_v{0.0f, 1.0f, 2.0f};
//! thrust::device_vector<char> char_v{'a', 'b', 'c'};
//!
//! cuda::zip_iterator iter{int_v.begin(), float_v.begin(), char_v.begin()};
//!
//! *iter;   // returns (0, 0.0f, 'a')
//! iter[0]; // returns (0, 0.0f, 'a')
//! iter[1]; // returns (1, 1.0f, 'b')
//! iter[2]; // returns (2, 2.0f, 'c')
//!
//! cuda::std::get<0>(iter[2]); // returns 2
//! cuda::std::get<1>(iter[0]); // returns 0.0f
//! cuda::std::get<2>(iter[1]); // returns 'b'
//!
//! // iter[3] is an out-of-bounds error
//! @endcode
//!
//! This example shows how to use @c zip_iterator to copy multiple ranges with a single call to @c thrust::copy.
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!   thrust::device_vector<int> int_in{0, 1, 2}, int_out(3);
//!   thrust::device_vector<float> float_in{0.0f, 10.0f, 20.0f}, float_out(3);
//!
//!   thrust::copy(cuda::zip_iterator{int_in.begin(), float_in.begin()},
//!                cuda::zip_iterator{int_in.end(),   float_in.end()},
//!                cuda::zip_iterator{int_out.begin(),float_out.begin()});
//!
//!   // int_out is now [0, 1, 2]
//!   // float_out is now [0.0f, 10.0f, 20.0f]
//!
//!   return 0;
//! }
//! @endcode
template <class... _Iterators>
class zip_iterator : public __zv_iter_category_base<_Iterators...>
{
  ::cuda::std::tuple<_Iterators...> __current_;

  template <class...>
  friend class zip_iterator;

  template <class _Fn>
  _CCCL_API static constexpr auto
  __zip_apply(const _Fn& __fun,
              const ::cuda::std::tuple<_Iterators...>& __tuple1,
              const ::cuda::std::tuple<_Iterators...>& __tuple2) //
    noexcept(noexcept(__fun(__tuple1, __tuple2, ::cuda::std::make_index_sequence<sizeof...(_Iterators)>())))
  {
    return __fun(__tuple1, __tuple2, ::cuda::std::make_index_sequence<sizeof...(_Iterators)>());
  }

public:
  //! @brief Default-constructs a @c zip_iterator by defaulting all stored iterators
  _CCCL_HIDE_FROM_ABI zip_iterator() = default;

  //! @brief Constructs a @c zip_iterator from a tuple of iterators
  //! @param __iters A tuple of iterators
  _CCCL_API constexpr explicit zip_iterator(::cuda::std::tuple<_Iterators...> __iters)
      : __current_(::cuda::std::move(__iters))
  {}

  //! @brief Constructs a @c zip_iterator from a tuple of iterators
  //! @param __iters A tuple of iterators
  _CCCL_TEMPLATE(size_t _NumIterators = sizeof...(_Iterators))
  _CCCL_REQUIRES((_NumIterators == 2))
  _CCCL_API constexpr explicit zip_iterator(::cuda::std::tuple<_Iterators...> __iters)
      : __current_(::cuda::std::get<0>(::cuda::std::move(__iters)), ::cuda::std::get<1>(::cuda::std::move(__iters)))
  {}

  //! @brief Constructs a @c zip_iterator from variadic set of iterators
  //! @param __iters The input iterators
  _CCCL_API constexpr explicit zip_iterator(_Iterators... __iters)
      : __current_(::cuda::std::move(__iters)...)
  {}

  using iterator_concept = decltype(__get_zip_iterator_concept<_Iterators...>());
  using value_type       = ::cuda::std::tuple<::cuda::std::iter_value_t<_Iterators>...>;
  using reference        = ::cuda::std::tuple<::cuda::std::iter_reference_t<_Iterators>...>;
  using difference_type  = ::cuda::std::common_type_t<::cuda::std::iter_difference_t<_Iterators>...>;

  // Those are technically not to spec, but pre-ranges iterator_traits do not work properly with iterators that do not
  // define all 5 aliases, see https://en.cppreference.com/w/cpp/iterator/iterator_traits.html
  using pointer = void;

  template <class... _OtherIters>
  static constexpr bool __all_convertible =
    (::cuda::std::convertible_to<_OtherIters, _Iterators> && ...)
    && !(::cuda::std::is_same_v<_Iterators, _OtherIters> && ...);

  //! @brief Converts a different @c zip_iterator
  //! @param __iter The other @c zip_iterator
  _CCCL_TEMPLATE(class... _OtherIters)
  _CCCL_REQUIRES((sizeof...(_OtherIters) == sizeof...(_Iterators)) _CCCL_AND __all_convertible<_OtherIters...>)
  _CCCL_API constexpr zip_iterator(zip_iterator<_OtherIters...> __iter)
      : __current_(::cuda::std::move(__iter.__current_))
  {}

  //! @brief Dereferences the @c zip_iterator
  //! @returns A tuple of references obtained by referencing every stored iterator
  [[nodiscard]] _CCCL_API constexpr auto operator*() const
    noexcept(noexcept(::cuda::std::apply(__zip_op_star{}, __current_)))
  {
    return ::cuda::std::apply(__zip_op_star{}, __current_);
  }

  struct __zip_op_index
  {
    difference_type __n;

    _CCCL_EXEC_CHECK_DISABLE
    [[nodiscard]] _CCCL_API constexpr reference operator()(const _Iterators&... __iters) const
      noexcept(noexcept(reference{__iters[::cuda::std::iter_difference_t<_Iterators>(__n)]...}))
    {
      return reference{__iters[::cuda::std::iter_difference_t<_Iterators>(__n)]...};
    }
  };

  //! @brief Subscripts the @c zip_iterator with an offset
  //! @param __n The additional offset
  //! @returns A tuple of references obtained by subscripting every stored iterator
  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_random_access)
  _CCCL_API constexpr auto operator[](difference_type __n) const
    noexcept(noexcept(::cuda::std::apply(__zip_op_index{__n}, __current_)))
  {
    return ::cuda::std::apply(__zip_op_index{__n}, __current_);
  }

  //! @brief Increments all stored iterators
  _CCCL_API constexpr zip_iterator& operator++() noexcept(noexcept(::cuda::std::apply(__zip_op_increment{}, __current_)))
  {
    ::cuda::std::apply(__zip_op_increment{}, __current_);
    return *this;
  }

  //! @brief Increments all stored iterators
  //! @returns A copy of the original @c zip_iterator if possible
  _CCCL_API constexpr auto operator++(int)
  {
    if constexpr (__zip_iter_constraints<_Iterators...>::__all_forward)
    {
      auto __tmp = *this;
      ++*this;
      return __tmp;
    }
    else
    {
      ++*this;
    }
  }

  //! @brief Decrements all stored iterators
  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_bidirectional)
  _CCCL_API constexpr zip_iterator& operator--() noexcept(noexcept(::cuda::std::apply(__zip_op_decrement{}, __current_)))
  {
    ::cuda::std::apply(__zip_op_decrement{}, __current_);
    return *this;
  }

  //! @brief Decrements all stored iterators
  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_bidirectional)
  _CCCL_API constexpr zip_iterator operator--(int)
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  struct __zip_op_pe
  {
    difference_type __n;

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_API constexpr void operator()(_Iterators&... __iters) const
      noexcept(noexcept(((void) (__iters += ::cuda::std::iter_difference_t<_Iterators>(__n)), ...)))
    {
      ((void) (__iters += ::cuda::std::iter_difference_t<_Iterators>(__n)), ...);
    }
  };

  //! @brief Increments all stored iterators by a given number of elements
  //! @param __n The number of elements to increment
  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_random_access)
  _CCCL_API constexpr zip_iterator&
  operator+=(difference_type __n) noexcept(noexcept(::cuda::std::apply(__zip_op_pe{__n}, __current_)))
  {
    ::cuda::std::apply(__zip_op_pe{__n}, __current_);
    return *this;
  }

  struct __zip_op_me
  {
    difference_type __n;

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_API constexpr void operator()(_Iterators&... __iters) const
      noexcept(noexcept(((void) (__iters -= ::cuda::std::iter_difference_t<_Iterators>(__n)), ...)))
    {
      ((void) (__iters -= ::cuda::std::iter_difference_t<_Iterators>(__n)), ...);
    }
  };

  //! @brief Decrements all stored iterators by a given number of elements
  //! @param __n The number of elements to decrement
  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_random_access)
  _CCCL_API constexpr zip_iterator&
  operator-=(difference_type __n) noexcept(noexcept(::cuda::std::apply(__zip_op_me{__n}, __current_)))
  {
    ::cuda::std::apply(__zip_op_me{__n}, __current_);
    return *this;
  }

  //! @brief Returns a copy of a @c zip_iterator incremented by a given number of elements
  //! @param __iter The @c zip_iterator to increment
  //! @param __n The number of elements to increment
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator+(const zip_iterator& __iter, difference_type __n)
    _CCCL_TRAILING_REQUIRES(zip_iterator)(_Constraints::__all_random_access)
  {
    auto __rhs = __iter;
    __rhs += __n;
    return __rhs;
  }

  //! @brief Returns a copy of a @c zip_iterator incremented by a given number of elements
  //! @param __n The number of elements to increment
  //! @param __iter The @c zip_iterator to increment
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator+(difference_type __n, const zip_iterator& __iter)
    _CCCL_TRAILING_REQUIRES(zip_iterator)(_Constraints::__all_random_access)
  {
    return __iter + __n;
  }

  //! @brief Returns a copy of a @c zip_iterator decremented by a given number of elements
  //! @param __n The number of elements to decrement
  //! @param __iter The @c zip_iterator to decrement
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator-(const zip_iterator& __iter, difference_type __n)
    _CCCL_TRAILING_REQUIRES(zip_iterator)(_Constraints::__all_random_access)
  {
    auto __rhs = __iter;
    __rhs -= __n;
    return __rhs;
  }

  struct __zip_op_minus
  {
    struct __less_abs
    {
      // abs in cstdlib is not constexpr
      _CCCL_EXEC_CHECK_DISABLE
      [[nodiscard]] _CCCL_API static constexpr difference_type
      __abs(difference_type __t) noexcept(noexcept(__t < 0 ? -__t : __t))
      {
        return __t < 0 ? -__t : __t;
      }

      _CCCL_EXEC_CHECK_DISABLE
      [[nodiscard]] _CCCL_API constexpr bool operator()(difference_type __n, difference_type __y) const
        noexcept(noexcept(__abs(__n) < __abs(__y)))
      {
        return __abs(__n) < __abs(__y);
      }
    };

    _CCCL_EXEC_CHECK_DISABLE
    template <size_t _Zero, size_t... _Indices>
    [[nodiscard]] _CCCL_API constexpr difference_type
    operator()(const ::cuda::std::tuple<_Iterators...>& __iters1,
               const ::cuda::std::tuple<_Iterators...>& __iters2,
               ::cuda::std::index_sequence<_Zero, _Indices...>) const //
      noexcept(noexcept(((::cuda::std::get<_Indices>(__iters1) - ::cuda::std::get<_Indices>(__iters2)) && ...)))
    {
      const auto __first = static_cast<difference_type>(::cuda::std::get<0>(__iters1) - ::cuda::std::get<0>(__iters2));
      if (__first == 0)
      {
        return __first;
      }

      const difference_type __temp[] = {
        __first,
        static_cast<difference_type>(::cuda::std::get<_Indices>(__iters1) - ::cuda::std::get<_Indices>(__iters2))...};
      return *::cuda::std::ranges::min_element(__temp, __zip_op_minus::__less_abs{});
    }
  };

  //! @brief Returns the distance between two @c zip_iterators
  //! @returns The minimal distance between any of the stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator-(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(difference_type)(_Constraints::__all_sized_sentinel)
  {
    return __zip_apply(__zip_op_minus{}, __n.__current_, __y.__current_);
  }

  struct __zip_op_eq
  {
    _CCCL_EXEC_CHECK_DISABLE
    template <size_t... _Indices>
    _CCCL_API constexpr bool operator()(const ::cuda::std::tuple<_Iterators...>& __iters1,
                                        const ::cuda::std::tuple<_Iterators...>& __iters2,
                                        ::cuda::std::index_sequence<_Indices...>) const
      noexcept(noexcept(((::cuda::std::get<_Indices>(__iters1) == ::cuda::std::get<_Indices>(__iters2)) || ...)))
    {
      return ((::cuda::std::get<_Indices>(__iters1) == ::cuda::std::get<_Indices>(__iters2)) || ...);
    }
  };

  //! @brief Compares two @c zip_iterator for equality by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator==(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_equality_comparable)
  {
    if constexpr (_Constraints::__all_bidirectional)
    {
      return __n.__current_ == __y.__current_;
    }
    else
    {
      return __zip_apply(__zip_op_eq{}, __n.__current_, __y.__current_);
    }
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two @c zip_iterator for inequality by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator!=(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_equality_comparable)
  {
    if constexpr (_Constraints::__all_bidirectional)
    {
      return __n.__current_ != __y.__current_;
    }
    else
    {
      return !__zip_apply(__zip_op_eq{}, __n.__current_, __y.__current_);
    }
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way compares two @c zip_iterator by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator<=>(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access&& _Constraints::__all_three_way_comparable)
  {
    return __n.__current_ <=> __y.__current_;
  }

#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv

  //! @brief Compares two @c zip_iterator for less than by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator<(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return __n.__current_ < __y.__current_;
  }

  //! @brief Compares two @c zip_iterator for greater than by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator>(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return __y < __n;
  }

  //! @brief Compares two @c zip_iterator for less equal by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator<=(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return !(__y < __n);
  }

  //! @brief Compares two @c zip_iterator for greater equal by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator>=(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return !(__n < __y);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  //! @brief Applies `iter_move` by applying it to all stored iterators
  // MSVC falls over its feet if this is not a template
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto
  iter_move(const zip_iterator& __iter) noexcept(_Constraints::__all_nothrow_iter_movable)
  {
    return ::cuda::std::apply(__zip_iter_move{}, __iter.__current_);
  }

  struct __zip_op_iter_swap
  {
    template <size_t... _Indices>
    _CCCL_API constexpr void operator()(const ::cuda::std::tuple<_Iterators...>& __iters1,
                                        const ::cuda::std::tuple<_Iterators...>& __iters2,
                                        ::cuda::std::index_sequence<_Indices...>) const
      noexcept(__zip_iter_constraints<_Iterators...>::__all_noexcept_swappable)
    {
      (::cuda::std::ranges::iter_swap(::cuda::std::get<_Indices>(__iters1), ::cuda::std::get<_Indices>(__iters2)), ...);
    }
  };

  //! @brief Applies `iter_swap` to two @c zip_iterator by applying it to all stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto
  iter_swap(const zip_iterator& __lhs, const zip_iterator& __rhs) noexcept(_Constraints::__all_noexcept_swappable)
    _CCCL_TRAILING_REQUIRES(void)(_Constraints::__all_indirectly_swappable)
  {
    return __zip_apply(__zip_op_iter_swap{}, __lhs.__current_, __rhs.__current_);
  }

  [[nodiscard]] _CCCL_API constexpr ::cuda::std::tuple<_Iterators...>& __iterators() noexcept
  {
    return __current_;
  }

  [[nodiscard]] _CCCL_API constexpr const ::cuda::std::tuple<_Iterators...>& __iterators() const noexcept
  {
    return __current_;
  }
};

template <class... _Iterators>
_CCCL_HOST_DEVICE zip_iterator(::cuda::std::tuple<_Iterators...>) -> zip_iterator<_Iterators...>;

template <class... _Iterators>
_CCCL_HOST_DEVICE zip_iterator(_Iterators...) -> zip_iterator<_Iterators...>;

//! @brief Creates a @c zip_iterator from a tuple of iterators.
//! @param __t The tuple of iterators to wrap
template <typename... Iterators>
_CCCL_API constexpr zip_iterator<Iterators...> make_zip_iterator(::cuda::std::tuple<Iterators...> __t)
{
  return zip_iterator<Iterators...>{::cuda::std::move(__t)};
}

//! @brief Creates a @c zip_iterator from a variadic number of iterators.
//! @param __iters The iterators to wrap
template <typename... Iterators>
_CCCL_API constexpr zip_iterator<Iterators...> make_zip_iterator(Iterators... __iters)
{
  return zip_iterator<Iterators...>{::cuda::std::move(__iters)...};
}

//! @}

_CCCL_END_NAMESPACE_CUDA

// GCC and MSVC2019 have issues determining __is_fancy_pointer in C++17 because they fail to instantiate pointer_traits
#if (_CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC)) && _CCCL_STD_VER <= 2017
_CCCL_BEGIN_NAMESPACE_CUDA_STD
template <class... _Iterators>
inline constexpr bool __is_fancy_pointer<::cuda::zip_iterator<_Iterators...>> = false;
_CCCL_END_NAMESPACE_CUDA_STD
#endif // _CCCL_COMPILER(MSVC) && _CCCL_STD_VER <= 2017

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_ZIP_ITERATOR_H
