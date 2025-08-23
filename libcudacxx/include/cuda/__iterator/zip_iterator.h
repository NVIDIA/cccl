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
#include <cuda/std/__utility/pair.h>
#include <cuda/std/tuple>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class... _Iterators>
struct __tuple_or_pair_impl
{
  using type = ::cuda::std::tuple<_Iterators...>;
};

template <class _Iterator1, class _Iterator2>
struct __tuple_or_pair_impl<_Iterator1, _Iterator2>
{
  using type = ::cuda::std::pair<_Iterator1, _Iterator2>;
};

template <class... _Iterators>
using __tuple_or_pair = typename __tuple_or_pair_impl<_Iterators...>::type;

template <class... _Iterators>
struct __zip_iter_constraints
{
  static constexpr bool __all_forward       = (::cuda::std::forward_iterator<_Iterators> && ...);
  static constexpr bool __all_bidirectional = (::cuda::std::bidirectional_iterator<_Iterators> && ...);
  static constexpr bool __all_random_access = (::cuda::std::random_access_iterator<_Iterators> && ...);

  static constexpr bool __all_equality_comparable = (::cuda::std::equality_comparable<_Iterators> && ...);

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  static constexpr bool __all_three_way_comparable = (::cuda::std::three_way_comparable<_Iterators> && ...);
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  static constexpr bool __all_sized_sentinel = (::cuda::std::sized_sentinel_for<_Iterators, _Iterators> && ...);
  static constexpr bool __all_nothrow_iter_movable =
    (noexcept(::cuda::std::ranges::iter_move(::cuda::std::declval<const _Iterators&>())) && ...)
    && (::cuda::std::is_nothrow_move_constructible_v<::cuda::std::iter_rvalue_reference_t<_Iterators>> && ...);

  static constexpr bool __all_indirectly_swappable = (::cuda::std::indirectly_swappable<_Iterators> && ...);

  static constexpr bool __all_noexcept_swappable = (::cuda::std::__noexcept_swappable<_Iterators> && ...);
};

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

template <class... _Iterators>
_CCCL_API constexpr auto __get_zip_view_iterator_tag()
{
  using _Constraints = __zip_iter_constraints<_Iterators...>;
  if constexpr (_Constraints::__all_random_access)
  {
    return ::cuda::std::random_access_iterator_tag();
  }
  else if constexpr (_Constraints::__all_bidirectional)
  {
    return ::cuda::std::bidirectional_iterator_tag();
  }
  else if constexpr (_Constraints::__all_forward)
  {
    return ::cuda::std::forward_iterator_tag();
  }
  else
  {
    return ::cuda::std::input_iterator_tag();
  }
  _CCCL_UNREACHABLE();
}

template <class... _Iterators>
class zip_iterator : public __zv_iter_category_base<_Iterators...>
{
  __tuple_or_pair<_Iterators...> __current_;

  template <class...>
  friend class zip_iterator;

  template <class _Fn>
  _CCCL_API static constexpr auto
  __zip_apply(const _Fn& __fun,
              const __tuple_or_pair<_Iterators...>& __tuple1,
              const __tuple_or_pair<_Iterators...>& __tuple2) //
    noexcept(noexcept(__fun(__tuple1, __tuple2, ::cuda::std::make_index_sequence<sizeof...(_Iterators)>())))
  {
    return __fun(__tuple1, __tuple2, ::cuda::std::make_index_sequence<sizeof...(_Iterators)>());
  }

public:
  _CCCL_API constexpr explicit zip_iterator(__tuple_or_pair<_Iterators...> __iters)
      : __current_(::cuda::std::move(__iters))
  {}

  // We want to use the simpler `pair` if possible, but still want to be able to construct from a 2 element tuple
  _CCCL_TEMPLATE(size_t _NumIterators = sizeof...(_Iterators))
  _CCCL_REQUIRES((_NumIterators == 2))
  _CCCL_API constexpr explicit zip_iterator(::cuda::std::tuple<_Iterators...> __iters)
      : __current_(::cuda::std::get<0>(::cuda::std::move(__iters)), ::cuda::std::get<1>(::cuda::std::move(__iters)))
  {}

  _CCCL_API constexpr explicit zip_iterator(_Iterators... __iters)
      : __current_(::cuda::std::move(__iters)...)
  {}

  using iterator_concept = decltype(__get_zip_view_iterator_tag<_Iterators...>());
  using value_type       = __tuple_or_pair<::cuda::std::iter_value_t<_Iterators>...>;
  using reference        = __tuple_or_pair<::cuda::std::iter_reference_t<_Iterators>...>;
  using difference_type  = ::cuda::std::common_type_t<::cuda::std::iter_difference_t<_Iterators>...>;

  _CCCL_HIDE_FROM_ABI zip_iterator() = default;

  template <class... _OtherIters>
  static constexpr bool __all_convertible =
    (::cuda::std::convertible_to<_OtherIters, _Iterators> && ...)
    && !(::cuda::std::is_same_v<_Iterators, _OtherIters> && ...);

  _CCCL_TEMPLATE(class... _OtherIters)
  _CCCL_REQUIRES((sizeof...(_OtherIters) == sizeof...(_Iterators)) _CCCL_AND __all_convertible<_OtherIters...>)
  _CCCL_API constexpr zip_iterator(zip_iterator<_OtherIters...> __i)
      : __current_(::cuda::std::move(__i.__current_))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API static constexpr reference
  __zip_op_star(const _Iterators&... __iters) noexcept(noexcept(reference{*__iters...}))
  {
    return reference{*__iters...};
  }

  [[nodiscard]] _CCCL_API constexpr auto operator*() const
    noexcept(noexcept(::cuda::std::apply(__zip_op_star, __current_)))
  {
    return ::cuda::std::apply(__zip_op_star, __current_);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API static constexpr void __zip_op_increment(_Iterators&... __iters) noexcept(noexcept(((void) ++__iters, ...)))
  {
    ((void) ++__iters, ...);
  }

  _CCCL_API constexpr zip_iterator& operator++() noexcept(noexcept(::cuda::std::apply(__zip_op_increment, __current_)))
  {
    ::cuda::std::apply(__zip_op_increment, __current_);
    return *this;
  }

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

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API static constexpr void __zip_op_decrement(_Iterators&... __iters) noexcept(noexcept(((void) --__iters, ...)))
  {
    ((void) --__iters, ...);
  }

  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_bidirectional)
  _CCCL_API constexpr zip_iterator& operator--() noexcept(noexcept(::cuda::std::apply(__zip_op_decrement, __current_)))
  {
    ::cuda::std::apply(__zip_op_decrement, __current_);
    return *this;
  }

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

  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_random_access)
  _CCCL_API constexpr zip_iterator& operator+=(difference_type __n)
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

  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_random_access)
  _CCCL_API constexpr zip_iterator& operator-=(difference_type __n)
  {
    ::cuda::std::apply(__zip_op_me{__n}, __current_);
    return *this;
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

  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_random_access)
  _CCCL_API constexpr auto operator[](difference_type __n) const
  {
    return ::cuda::std::apply(__zip_op_index{__n}, __current_);
  }

  struct __zip_op_eq
  {
    _CCCL_EXEC_CHECK_DISABLE
    template <size_t... _Indices>
    _CCCL_API constexpr bool operator()(const __tuple_or_pair<_Iterators...>& __iters1,
                                        const __tuple_or_pair<_Iterators...>& __iters2,
                                        ::cuda::std::index_sequence<_Indices...>) const
      noexcept(noexcept(((::cuda::std::get<_Indices>(__iters1) == ::cuda::std::get<_Indices>(__iters2)) || ...)))
    {
      return ((::cuda::std::get<_Indices>(__iters1) == ::cuda::std::get<_Indices>(__iters2)) || ...);
    }
  };

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
    _CCCL_UNREACHABLE();
  }

#if _CCCL_STD_VER <= 2017
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
    _CCCL_UNREACHABLE();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator<=>(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access&& _Constraints::__all_three_way_comparable)
  {
    return __n.__current_ <=> __y.__current_;
  }

#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator<(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return __n.__current_ < __y.__current_;
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator>(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return __y < __n;
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator<=(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return !(__y < __n);
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator>=(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return !(__n < __y);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator+(const zip_iterator& __i, difference_type __n)
    _CCCL_TRAILING_REQUIRES(zip_iterator)(_Constraints::__all_random_access)
  {
    auto __r = __i;
    __r += __n;
    return __r;
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator+(difference_type __n, const zip_iterator& __i)
    _CCCL_TRAILING_REQUIRES(zip_iterator)(_Constraints::__all_random_access)
  {
    return __i + __n;
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator-(const zip_iterator& __i, difference_type __n)
    _CCCL_TRAILING_REQUIRES(zip_iterator)(_Constraints::__all_random_access)
  {
    auto __r = __i;
    __r -= __n;
    return __r;
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
    operator()(const __tuple_or_pair<_Iterators...>& __iters1,
               const __tuple_or_pair<_Iterators...>& __iters2,
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

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto operator-(const zip_iterator& __n, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(difference_type)(_Constraints::__all_sized_sentinel)
  {
    return __zip_apply(__zip_op_minus{}, __n.__current_, __y.__current_);
  }

  using __iter_move_ret = __tuple_or_pair<::cuda::std::iter_rvalue_reference_t<_Iterators>...>;

  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API static constexpr __iter_move_ret __zip_iter_move(const _Iterators&... __iters) noexcept(
    noexcept(__iter_move_ret{::cuda::std::ranges::iter_move(__iters)...}))
  {
    return __iter_move_ret{::cuda::std::ranges::iter_move(__iters)...};
  }

  // MSVC falls over its feet if this is not a template
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto iter_move(const zip_iterator& __i) noexcept(_Constraints::__all_nothrow_iter_movable)
  {
    return ::cuda::std::apply(__zip_iter_move, __i.__current_);
  }

  template <class... _OtherIterators>
  static constexpr bool __all_nothrow_swappable =
    (::cuda::std::__noexcept_swappable<_OtherIterators, _OtherIterators> && ...);

  struct __zip_op_iter_swap
  {
    template <size_t... _Indices>
    _CCCL_API constexpr void
    operator()(const __tuple_or_pair<_Iterators...>& __iters1,
               const __tuple_or_pair<_Iterators...>& __iters2,
               ::cuda::std::index_sequence<_Indices...>) const noexcept(__all_nothrow_swappable<_Iterators...>)
    {
      (::cuda::std::ranges::iter_swap(::cuda::std::get<_Indices>(__iters1), ::cuda::std::get<_Indices>(__iters2)), ...);
    }
  };

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  _CCCL_API friend constexpr auto
  iter_swap(const zip_iterator& __l, const zip_iterator& __r) noexcept(_Constraints::__all_noexcept_swappable)
    _CCCL_TRAILING_REQUIRES(void)(_Constraints::__all_indirectly_swappable)
  {
    return __zip_apply(__zip_op_iter_swap{}, __l.__current_, __r.__current_);
  }
};

template <class... _Iterators>
_CCCL_HOST_DEVICE zip_iterator(::cuda::std::tuple<_Iterators...>) -> zip_iterator<_Iterators...>;

template <class _Iterator1, class _Iterator2>
_CCCL_HOST_DEVICE zip_iterator(::cuda::std::pair<_Iterator1, _Iterator2>) -> zip_iterator<_Iterator1, _Iterator2>;

template <class... _Iterators>
_CCCL_HOST_DEVICE zip_iterator(_Iterators...) -> zip_iterator<_Iterators...>;

//! \p make_zip_iterator creates a \p zip_iterator from a \p tuple of iterators.
//!
//! \param t The \p tuple of iterators to copy.
//! \return A newly created \p zip_iterator which zips the iterators encapsulated in \p t.
template <typename... Iterators>
_CCCL_API constexpr zip_iterator<Iterators...> make_zip_iterator(::cuda::std::tuple<Iterators...> t)
{
  return zip_iterator<Iterators...>{::cuda::std::move(t)};
}

//! \p make_zip_iterator creates a \p zip_iterator from iterators.
//!
//! \param its The iterators to copy.
//! \return A newly created \p zip_iterator which zips the iterators.
template <typename... Iterators>
_CCCL_API constexpr zip_iterator<Iterators...> make_zip_iterator(Iterators... its)
{
  return zip_iterator<Iterators...>{::cuda::std::move(its)...};
}

_CCCL_END_NAMESPACE_CUDA

// GCC and MSVC2019 have issues determining _IsFancyPointer in C++17 because they fail to instantiate pointer_traits
#if (_CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC)) && _CCCL_STD_VER <= 2017
_CCCL_BEGIN_NAMESPACE_CUDA_STD
template <class... _Iterators>
struct _IsFancyPointer<::cuda::zip_iterator<_Iterators...>> : false_type
{};
_CCCL_END_NAMESPACE_CUDA_STD
#endif // _CCCL_COMPILER(MSVC) && _CCCL_STD_VER <= 2017

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_ZIP_ITERATOR_H
