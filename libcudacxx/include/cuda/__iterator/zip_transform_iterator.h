// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___ITERATOR_ZIP_TRANSFORM_ITERATOR_H
#define _CUDA___ITERATOR_ZIP_TRANSFORM_ITERATOR_H

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
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/compressed_movable_box.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/movable_box.h>
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

template <class _Fn, class... _Iterators>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __get_zip_transform_iterator_category()
{
  using _Constraints = __zip_iter_constraints<_Iterators...>;
  if constexpr (!::cuda::std::is_reference_v<
                  ::cuda::std::invoke_result_t<_Fn&, ::cuda::std::iter_reference_t<_Iterators>...>>)
  {
    return ::cuda::std::input_iterator_tag{};
  }
  else if constexpr (_Constraints::__all_random_access)
  {
    return ::cuda::std::random_access_iterator_tag{};
  }
  else if constexpr (_Constraints::__all_bidirectional)
  {
    return ::cuda::std::bidirectional_iterator_tag{};
  }
  else if constexpr (_Constraints::__all_forward)
  {
    return ::cuda::std::forward_iterator_tag{};
  }
  else
  {
    return ::cuda::std::input_iterator_tag{};
  }
}

//! @brief @c zip_transform_iterator is an iterator which represents the result of a transformation of a set of
//! sequences with a given function. This iterator is useful for creating a range filled with the result of applying an
//! operation to another range without either explicitly storing it in memory, or explicitly executing the
//! transformation. Using @c zip_transform_iterator facilitates kernel fusion by deferring the execution of a
//! transformation until the value is needed while saving both memory capacity and bandwidth.
//!
//! @c zip_transform_iterator is morally equivalent to a combination of transform_iterator and zip_iterator
//!
//! @code{.cpp}
//!   template <class Fn, class... Iterators>
//!   using zip_transform_iterator = cuda::transform_iterator<cuda::zip_iterator<Iterators...>, cuda::zip_function<Fn>>;
//! @endcode
//!
//! @c zip_transform_iterator has the additional benefit that it does not require an artificial @c zip_function to work
//! and more importantly does not need to materialize the result of dereferencing the stored iterators when passing them
//! to the stored function.
//!
//! The following code snippet demonstrates how to create a @c zip_transform_iterator which represents the result of
//! "zipping" multiple ranges together.
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//!
//! struct SumArgs {
//!   __host__ __device__ float operator()(float a, float b, float c) const noexcept {
//!     return a + b + c;
//!   }
//! };
//!
//! thrust::device_vector<float> A{0.f, 1.f, 2.f};
//! thrust::device_vector<float> B{1.f, 2.f, 3.f};
//! thrust::device_vector<float> C{2.f, 3.f, 4.f};
//!
//! cuda::zip_transform_iterator iter{SumArgs{}, A.begin(), B.begin(), C.begin()};
//!
//! *iter;   // returns (3.f)
//! iter[0]; // returns (3.f)
//! iter[1]; // returns (6.f)
//! iter[2]; // returns (9.f)
//! // iter[3] is an out-of-bounds error
//! @endcode
//!
//! This example shows how to use @c zip_transform_iterator to copy multiple ranges with a single call to @c
//! thrust::copy.
//!
//! @code
//! #include <cuda/iterator>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!   struct SumArgs {
//!     __host__ __device__ float operator()(float a, float b, float c) const noexcept {
//!       return a + b + c;
//!     }
//!   };
//!
//!   thrust::device_vector<float> A{0.f, 1.f, 2.f};
//!   thrust::device_vector<float> B{1.f, 2.f, 3.f};
//!   thrust::device_vector<float> C{2.f, 3.f, 4.f};
//!   thrust::device_vector<float> out(3);
//!
//!   cuda::zip_transform_iterator iter{SumArgs{}, A.begin(), B.begin(), C.begin()}
//!   thrust::copy(iter, iter + 3, out.begin());
//!
//!   // out is now [3.0f, 6.0f, 9.0f]
//!
//!   return 0;
//! }
//! @endcode
template <class _Fn, class... _Iterators>
class zip_transform_iterator
{
private:
  // Not a base because then the friend operators would be ambiguous
  ::cuda::std::__compressed_movable_box<::cuda::std::tuple<_Iterators...>, _Fn> __store_;

  [[nodiscard]] _CCCL_API constexpr ::cuda::std::tuple<_Iterators...>& __iters() noexcept
  {
    return __store_.template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr const ::cuda::std::tuple<_Iterators...>& __iters() const noexcept
  {
    return __store_.template __get<0>();
  }

  [[nodiscard]] _CCCL_API constexpr _Fn& __func() noexcept
  {
    return __store_.template __get<1>();
  }

  [[nodiscard]] _CCCL_API constexpr const _Fn& __func() const noexcept
  {
    return __store_.template __get<1>();
  }

  template <class, class...>
  friend class zip_transform_iterator;

  template <class _Op>
  _CCCL_API static constexpr auto
  __zip_apply(const _Op& __op,
              const ::cuda::std::tuple<_Iterators...>& __tuple1,
              const ::cuda::std::tuple<_Iterators...>& __tuple2) //
    noexcept(noexcept(__op(__tuple1, __tuple2, ::cuda::std::make_index_sequence<sizeof...(_Iterators)>())))
  {
    return __op(__tuple1, __tuple2, ::cuda::std::make_index_sequence<sizeof...(_Iterators)>());
  }

public:
  //! @brief Default-constructs a @c zip_transform_iterator by value-initializing the functor and all stored iterators
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Fn2 = _Fn)
  _CCCL_REQUIRES(
    ::cuda::std::default_initializable<_Fn2>&& __zip_iter_constraints<_Iterators...>::__all_default_initializable)
  _CCCL_API constexpr zip_transform_iterator() noexcept(
    ::cuda::std::is_nothrow_default_constructible_v<_Fn2>
    && __zip_iter_constraints<_Iterators...>::__all_nothrow_default_constructible)
      : __store_()
  {}

  //! @brief Constructs a @c zip_transform_iterator from a tuple of iterators
  //! @param __iters A tuple or pair of iterators
  _CCCL_API constexpr explicit zip_transform_iterator(_Fn __fun, ::cuda::std::tuple<_Iterators...> __iters)
      : __store_(::cuda::std::move(__iters), ::cuda::std::move(__fun))
  {}

  //! @brief Constructs a @c zip_transform_iterator from variadic set of iterators
  //! @param __iters The input iterators
  _CCCL_API constexpr explicit zip_transform_iterator(_Fn __fun, _Iterators... __iters)
      : __store_(::cuda::std::tuple<_Iterators...>{::cuda::std::move(__iters)...}, ::cuda::std::move(__fun))
  {}

  using iterator_concept  = decltype(::cuda::__get_zip_iterator_concept<_Iterators...>());
  using iterator_category = decltype(::cuda::__get_zip_transform_iterator_category<_Fn, _Iterators...>());
  using difference_type   = ::cuda::std::common_type_t<::cuda::std::iter_difference_t<_Iterators>...>;
  using value_type =
    ::cuda::std::remove_cvref_t<::cuda::std::invoke_result_t<_Fn&, ::cuda::std::iter_reference_t<_Iterators>...>>;

  // Those are technically not to spec, but pre-ranges iterator_traits do not work properly with iterators that do not
  // define all 5 aliases, see https://en.cppreference.com/w/cpp/iterator/iterator_traits.html
  using reference = ::cuda::std::invoke_result_t<_Fn&, ::cuda::std::iter_reference_t<_Iterators>...>;
  using pointer   = void;

  // Internal helper functions to extract internals for device dispatch, must be a tuple for cub_transform_many
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::tuple<_Iterators...>
  __base() && noexcept(::cuda::std::is_nothrow_move_constructible_v<::cuda::std::tuple<_Iterators...>>)
  {
    return ::cuda::std::move(__iters());
  }

  [[nodiscard]] _CCCL_API constexpr _Fn __pred() && noexcept(::cuda::std::is_nothrow_move_constructible_v<_Fn>)
  {
    return ::cuda::std::move(__func());
  }

  struct __zip_transform_op_star
  {
    _Fn& __func_;

    _CCCL_EXEC_CHECK_DISABLE
    [[nodiscard]] _CCCL_API constexpr reference operator()(const _Iterators&... __iters) const
      noexcept(::cuda::std::is_nothrow_invocable_v<_Fn&, ::cuda::std::iter_reference_t<const _Iterators>...>)
    {
      return ::cuda::std::invoke(const_cast<_Fn&>(__func_), *__iters...);
    }
  };

  //! @brief Invokes the stored function with the result of dereferencing the stored iterators
  [[nodiscard]] _CCCL_API constexpr reference operator*() const
    noexcept(::cuda::std::is_nothrow_invocable_v<_Fn&, ::cuda::std::iter_reference_t<const _Iterators>...>)
  {
    return ::cuda::std::apply(__zip_transform_op_star{const_cast<_Fn&>(__func())}, __iters());
  }

  struct __zip_transform_op_subscript
  {
    difference_type __n_;
    _Fn& __func_;

    _CCCL_EXEC_CHECK_DISABLE
    [[nodiscard]] _CCCL_API constexpr reference operator()(const _Iterators&... __iters) const noexcept(noexcept(
      ::cuda::std::invoke(const_cast<_Fn&>(__func_), __iters[::cuda::std::iter_difference_t<_Iterators>(__n_)]...)))
    {
      return ::cuda::std::invoke(
        const_cast<_Fn&>(__func_), __iters[::cuda::std::iter_difference_t<_Iterators>(__n_)]...);
    }
  };

  //! @brief Invokes the stored function with the result of dereferencing the stored iterators advanced by an offset
  //! @param __n The additional offset
  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_random_access)
  _CCCL_API constexpr reference operator[](difference_type __n) const
    noexcept(noexcept(::cuda::std::apply(__zip_transform_op_subscript{__n, ::cuda::std::declval<_Fn&>()},
                                         ::cuda::std::declval<const ::cuda::std::tuple<_Iterators...>&>())))
  {
    return ::cuda::std::apply(__zip_transform_op_subscript{__n, const_cast<_Fn&>(__func())}, __iters());
  }

  //! @brief Increments all stored iterators
  _CCCL_API constexpr zip_transform_iterator& operator++() noexcept(
    noexcept(::cuda::std::apply(__zip_op_increment{}, ::cuda::std::declval<::cuda::std::tuple<_Iterators...>&>())))
  {
    ::cuda::std::apply(__zip_op_increment{}, __iters());
    return *this;
  }

  //! @brief Increments all stored iterators
  //! @returns A copy of the original @c zip_transform_iterator if possible
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
  _CCCL_API constexpr zip_transform_iterator& operator--() noexcept(
    noexcept(::cuda::std::apply(__zip_op_decrement{}, ::cuda::std::declval<::cuda::std::tuple<_Iterators...>&>())))
  {
    ::cuda::std::apply(__zip_op_decrement{}, __iters());
    return *this;
  }

  //! @brief Decrements all stored iterators
  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_bidirectional)
  _CCCL_API constexpr zip_transform_iterator operator--(int)
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
  _CCCL_API constexpr zip_transform_iterator& operator+=(difference_type __n) noexcept(
    noexcept(::cuda::std::apply(__zip_op_pe{__n}, ::cuda::std::declval<::cuda::std::tuple<_Iterators...>&>())))
  {
    ::cuda::std::apply(__zip_op_pe{__n}, __iters());
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
  _CCCL_API constexpr zip_transform_iterator& operator-=(difference_type __n) noexcept(
    noexcept(::cuda::std::apply(__zip_op_me{__n}, ::cuda::std::declval<::cuda::std::tuple<_Iterators...>&>())))
  {
    ::cuda::std::apply(__zip_op_me{__n}, __iters());
    return *this;
  }

  //! @brief Returns a copy of a @c zip_transform_iterator incremented by a given number of elements
  //! @param __iter The @c zip_transform_iterator to increment
  //! @param __n The number of elements to increment
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator+(const zip_transform_iterator& __iter, difference_type __n)
    _CCCL_TRAILING_REQUIRES(zip_transform_iterator)(_Constraints::__all_random_access)
  {
    auto __rhs = __iter;
    __rhs += __n;
    return __rhs;
  }

  //! @brief Returns a copy of a @c zip_transform_iterator incremented by a given number of elements
  //! @param __n The number of elements to increment
  //! @param __iter The @c zip_transform_iterator to increment
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator+(difference_type __n, const zip_transform_iterator& __iter)
    _CCCL_TRAILING_REQUIRES(zip_transform_iterator)(_Constraints::__all_random_access)
  {
    return __iter + __n;
  }

  //! @brief Returns a copy of a @c zip_transform_iterator decremented by a given number of elements
  //! @param __n The number of elements to decrement
  //! @param __iter The @c zip_transform_iterator to decrement
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator-(const zip_transform_iterator& __iter, difference_type __n)
    _CCCL_TRAILING_REQUIRES(zip_transform_iterator)(_Constraints::__all_random_access)
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

  //! @brief Returns the distance between two @c zip_transform_iterators
  //! @returns The minimal distance between any of the stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator-(const zip_transform_iterator& __n, const zip_transform_iterator& __y)
    _CCCL_TRAILING_REQUIRES(difference_type)(_Constraints::__all_sized_sentinel)
  {
    return __zip_apply(__zip_op_minus{}, __n.__iters(), __y.__iters());
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

  //! @brief Compares two @c zip_transform_iterator for equality by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator==(const zip_transform_iterator& __n, const zip_transform_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_equality_comparable)
  {
    if constexpr (_Constraints::__all_bidirectional)
    {
      return __n.__iters() == __y.__iters();
    }
    else
    {
      return __zip_apply(__zip_op_eq{}, __n.__iters(), __y.__iters());
    }
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Compares two @c zip_transform_iterator for inequality by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator!=(const zip_transform_iterator& __n, const zip_transform_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_equality_comparable)
  {
    if constexpr (_Constraints::__all_bidirectional)
    {
      return __n.__iters() != __y.__iters();
    }
    else
    {
      return !__zip_apply(__zip_op_eq{}, __n.__iters(), __y.__iters());
    }
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  //! @brief Three-way compares two @c zip_transform_iterator by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator<=>(const zip_transform_iterator& __n, const zip_transform_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access&& _Constraints::__all_three_way_comparable)
  {
    return __n.__iters() <=> __y.__iters();
  }

#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv

  //! @brief Compares two @c zip_transform_iterator for less than by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator<(const zip_transform_iterator& __n, const zip_transform_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return __n.__iters() < __y.__iters();
  }

  //! @brief Compares two @c zip_transform_iterator for greater than by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator>(const zip_transform_iterator& __n, const zip_transform_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return __y < __n;
  }

  //! @brief Compares two @c zip_transform_iterator for less equal by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator<=(const zip_transform_iterator& __n, const zip_transform_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return !(__y < __n);
  }

  //! @brief Compares two @c zip_transform_iterator for greater equal by comparing the tuple of stored iterators
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator>=(const zip_transform_iterator& __n, const zip_transform_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return !(__n < __y);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};

template <class _Fn, class... _Iterators>
_CCCL_HOST_DEVICE zip_transform_iterator(_Fn, ::cuda::std::tuple<_Iterators...>)
  -> zip_transform_iterator<_Fn, _Iterators...>;

template <class _Fn, class... _Iterators>
_CCCL_HOST_DEVICE zip_transform_iterator(_Fn, _Iterators...) -> zip_transform_iterator<_Fn, _Iterators...>;

//! @brief Creates a @c zip_transform_iterator from a tuple of iterators.
//! @param __t The tuple of iterators to wrap
template <class _Fn, class... _Iterators>
[[nodiscard]] _CCCL_API constexpr auto
make_zip_transform_iterator(_Fn __fun, ::cuda::std::tuple<_Iterators...> __t) noexcept(
  ::cuda::std::is_nothrow_move_constructible_v<_Fn>
  && __zip_iter_constraints<_Iterators...>::__all_nothrow_move_constructible)
{
  return zip_transform_iterator<_Fn, _Iterators...>{::cuda::std::move(__fun), ::cuda::std::move(__t)};
}

//! @brief Creates a @c zip_transform_iterator from a variadic number of iterators.
//! @param __iters The iterators to wrap
template <class _Fn, class... _Iterators>
[[nodiscard]] _CCCL_API constexpr auto make_zip_transform_iterator(_Fn __fun, _Iterators... __iters) noexcept(
  ::cuda::std::is_nothrow_move_constructible_v<_Fn>
  && __zip_iter_constraints<_Iterators...>::__all_nothrow_move_constructible)
{
  return zip_transform_iterator<_Fn, _Iterators...>{::cuda::std::move(__fun), ::cuda::std::move(__iters)...};
}

//! @}

_CCCL_END_NAMESPACE_CUDA

// GCC and MSVC2019 have issues determining __is_fancy_pointer in C++17 because they fail to instantiate pointer_traits
#if (_CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC)) && _CCCL_STD_VER <= 2017
_CCCL_BEGIN_NAMESPACE_CUDA_STD
template <class _Fn, class... _Iterators>
inline constexpr bool __is_fancy_pointer<::cuda::zip_transform_iterator<_Fn, _Iterators...>> = false;
_CCCL_END_NAMESPACE_CUDA_STD
#endif // (_CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC)) && _CCCL_STD_VER <= 2017

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_ZIP_TRANSFORM_ITERATOR_H
