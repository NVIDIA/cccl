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

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class... _Iterators>
struct __tuple_or_pair_impl
{
  using type = _CUDA_VSTD::tuple<_Iterators...>;
};

template <class _Iterator1, class _Iterator2>
struct __tuple_or_pair_impl<_Iterator1, _Iterator2>
{
  using type = _CUDA_VSTD::pair<_Iterator1, _Iterator2>;
};

template <class... _Iterators>
using __tuple_or_pair = typename __tuple_or_pair_impl<_Iterators...>::type;

struct __zip_iter_functors
{
  // iterator functions
  template <class _Tuple1, class _Tuple2, size_t... _Indices>
  _CCCL_API static constexpr bool
  __iter_op_eq(const _Tuple1& __tuple1, const _Tuple2& __tuple2, _CUDA_VSTD::index_sequence<_Indices...>) noexcept(
    noexcept(((_CUDA_VSTD::get<_Indices>(__tuple1) == _CUDA_VSTD::get<_Indices>(__tuple2)) || ...)))
  {
    return ((_CUDA_VSTD::get<_Indices>(__tuple1) == _CUDA_VSTD::get<_Indices>(__tuple2)) || ...);
  }

  template <class _Tuple1, class _Tuple2>
  _CCCL_API static constexpr bool
  __iter_op_eq(const _Tuple1& __tuple1, const _Tuple2& __tuple2) noexcept(noexcept(__zip_iter_functors::__iter_op_eq(
    __tuple1,
    __tuple2,
    _CUDA_VSTD::make_index_sequence<_CUDA_VSTD::tuple_size_v<_CUDA_VSTD::remove_cvref_t<_Tuple1>>>())))
  {
    return __zip_iter_functors::__iter_op_eq(
      __tuple1,
      __tuple2,
      _CUDA_VSTD::make_index_sequence<_CUDA_VSTD::tuple_size_v<_CUDA_VSTD::remove_cvref_t<_Tuple1>>>());
  }

  struct __zip_op_star
  {
    struct __op_star
    {
      template <class _Iter>
      _CCCL_API constexpr decltype(auto) operator()(_Iter& __i) const noexcept(noexcept(*__i))
      {
        return *__i;
      }
    };

    template <class... _Types>
    _CCCL_API constexpr auto operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(__tuple_or_pair<_CUDA_VSTD::invoke_result_t<__op_star&, _Types>...>{
        _CUDA_VSTD::invoke(__op_star{}, _CUDA_VSTD::forward<_Types>(__tuple_elements))...}))
    {
      return __tuple_or_pair<_CUDA_VSTD::invoke_result_t<__op_star&, _Types>...>{
        _CUDA_VSTD::invoke(__op_star{}, _CUDA_VSTD::forward<_Types>(__tuple_elements))...};
    }
  };

  template <class _Tuple>
  _CCCL_API static constexpr auto __iter_op_star(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zip_iter_functors::__zip_op_star{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    return _CUDA_VSTD::apply(__zip_iter_functors::__zip_op_star{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  struct __zip_op_increment
  {
    struct __op_increment
    {
      template <class _Iter>
      _CCCL_API constexpr void operator()(_Iter& __i) const noexcept(noexcept(++__i))
      {
        ++__i;
      }
    };

    template <class... _Types>
    _CCCL_API constexpr void operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept((_CUDA_VSTD::invoke(__op_increment{}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...)))
    {
      (_CUDA_VSTD::invoke(__op_increment{}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...);
    }
  };

  template <class _Tuple>
  _CCCL_API static constexpr void __iter_op_increment(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zip_iter_functors::__zip_op_increment{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    _CUDA_VSTD::apply(__zip_iter_functors::__zip_op_increment{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  struct __zip_op_decrement
  {
    struct __op_decrement
    {
      template <class _Iter>
      _CCCL_API constexpr void operator()(_Iter& __i) const noexcept(noexcept(--__i))
      {
        --__i;
      }
    };

    template <class... _Types>
    _CCCL_API constexpr void operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept((_CUDA_VSTD::invoke(__op_decrement{}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...)))
    {
      (_CUDA_VSTD::invoke(__op_decrement{}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...);
    }
  };

  template <class _Tuple>
  _CCCL_API static constexpr void __iter_op_decrement(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zip_iter_functors::__zip_op_decrement{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    _CUDA_VSTD::apply(__zip_iter_functors::__zip_op_decrement{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  template <class _Diff>
  struct __zip_op_pe
  {
    _Diff __x;

    struct __op_pe
    {
      _Diff __x;

      template <class _Iter>
      _CCCL_API constexpr void operator()(_Iter& __i) const
        noexcept(noexcept(__i += _CUDA_VSTD::iter_difference_t<_Iter>(__x)))
      {
        __i += _CUDA_VSTD::iter_difference_t<_Iter>(__x);
      }
    };

    template <class... _Types>
    _CCCL_API constexpr void operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept((_CUDA_VSTD::invoke(__op_pe{__x}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...)))
    {
      (_CUDA_VSTD::invoke(__op_pe{__x}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...);
    }
  };

  template <class _Diff, class _Tuple>
  _CCCL_API static constexpr void __iter_op_pe(_Diff __x, _Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zip_iter_functors::__zip_op_pe<_Diff>{__x}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    _CUDA_VSTD::apply(__zip_iter_functors::__zip_op_pe<_Diff>{__x}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  template <class _Diff>
  struct __zip_op_me
  {
    _Diff __x;

    struct __op_me
    {
      _Diff __x;

      template <class _Iter>
      _CCCL_API constexpr void operator()(_Iter& __i) const
        noexcept(noexcept(__i -= _CUDA_VSTD::iter_difference_t<_Iter>(__x)))
      {
        __i -= _CUDA_VSTD::iter_difference_t<_Iter>(__x);
      }
    };

    template <class... _Types>
    _CCCL_API constexpr void operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept((_CUDA_VSTD::invoke(__op_me{__x}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...)))
    {
      (_CUDA_VSTD::invoke(__op_me{__x}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...);
    }
  };

  template <class _Diff, class _Tuple>
  _CCCL_API static constexpr void __iter_op_me(_Diff __x, _Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zip_iter_functors::__zip_op_me<_Diff>{__x}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    _CUDA_VSTD::apply(__zip_iter_functors::__zip_op_me<_Diff>{__x}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  template <class _Diff>
  struct __zip_op_index
  {
    _Diff __n;

    struct __op_index
    {
      _Diff __n;

      template <class _Iter>
      _CCCL_API constexpr decltype(auto) operator()(_Iter& __i) const
        noexcept(noexcept(__i[_CUDA_VSTD::iter_difference_t<_Iter>(__n)]))
      {
        return __i[_CUDA_VSTD::iter_difference_t<_Iter>(__n)];
      }
    };

    template <class... _Types>
    _CCCL_API constexpr auto operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(__tuple_or_pair<_CUDA_VSTD::invoke_result_t<__zip_op_index::__op_index&, _Types>...>{
        _CUDA_VSTD::invoke(__zip_op_index::__op_index{__n}, _CUDA_VSTD::forward<_Types>(__tuple_elements))...}))
    {
      return __tuple_or_pair<_CUDA_VSTD::invoke_result_t<__zip_op_index::__op_index&, _Types>...>{
        _CUDA_VSTD::invoke(__zip_op_index::__op_index{__n}, _CUDA_VSTD::forward<_Types>(__tuple_elements))...};
    }
  };

  template <class _Diff, class _Tuple>
  _CCCL_API static constexpr auto __iter_op_index(_Diff __n, _Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zip_iter_functors::__zip_op_index<_Diff>{__n}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    return _CUDA_VSTD::apply(__zip_iter_functors::__zip_op_index<_Diff>{__n}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  struct __op_comp_abs
  {
    // abs in cstdlib is not constexpr
    template <class _Diff>
    _CCCL_API static constexpr _Diff __abs(_Diff __t) noexcept(noexcept(__t < 0 ? -__t : __t))
    {
      return __t < 0 ? -__t : __t;
    }

    template <class _Diff>
    _CCCL_API constexpr bool operator()(const _Diff& __x, const _Diff& __y) const
      noexcept(noexcept(__op_comp_abs::__abs(__x) < __op_comp_abs::__abs(__y)))
    {
      return __op_comp_abs::__abs(__x) < __op_comp_abs::__abs(__y);
    }
  };

  template <class _Diff, class _Tuple1, class _Tuple2, size_t _Zero, size_t... _Indices>
  _CCCL_API static constexpr _Diff
  __iter_op_minus(const _Tuple1& __tuple1, const _Tuple2& __tuple2, _CUDA_VSTD::index_sequence<_Zero, _Indices...>) //
    noexcept(noexcept(((_CUDA_VSTD::get<_Indices>(__tuple1) - _CUDA_VSTD::get<_Indices>(__tuple2)) && ...)))
  {
    const _Diff __first = _CUDA_VSTD::get<0>(__tuple1) - _CUDA_VSTD::get<0>(__tuple2);
    if (__first == 0)
    {
      return __first;
    }

    const _Diff __temp[] = {__first, _CUDA_VSTD::get<_Indices>(__tuple1) - _CUDA_VSTD::get<_Indices>(__tuple2)...};
    return *(_CUDA_VRANGES::min_element) (__temp, __op_comp_abs{});
  }

  template <class _Diff, class _Tuple1, class _Tuple2>
  _CCCL_API static constexpr _Diff __iter_op_minus(const _Tuple1& __tuple1, const _Tuple2& __tuple2) noexcept(
    noexcept(__zip_iter_functors::__iter_op_minus<_Diff>(
      __tuple1,
      __tuple2,
      _CUDA_VSTD::make_index_sequence<_CUDA_VSTD::tuple_size_v<_CUDA_VSTD::remove_cvref_t<_Tuple1>>>())))
  {
    return __zip_iter_functors::__iter_op_minus<_Diff>(
      __tuple1,
      __tuple2,
      _CUDA_VSTD::make_index_sequence<_CUDA_VSTD::tuple_size_v<_CUDA_VSTD::remove_cvref_t<_Tuple1>>>());
  }

  struct __zip_iter_move
  {
    template <class... _Types>
    _CCCL_API constexpr __tuple_or_pair<_CUDA_VSTD::invoke_result_t<decltype(_CUDA_VRANGES::iter_move)&, _Types>...>
    operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(__tuple_or_pair<_CUDA_VSTD::invoke_result_t<decltype(_CUDA_VRANGES::iter_move)&, _Types>...>{
        _CUDA_VSTD::invoke(_CUDA_VRANGES::iter_move, _CUDA_VSTD::forward<_Types>(__tuple_elements))...}))
    {
      return {_CUDA_VSTD::invoke(_CUDA_VRANGES::iter_move, _CUDA_VSTD::forward<_Types>(__tuple_elements))...};
    }
  };

  template <class _Tuple>
  _CCCL_API static constexpr auto __iter_move(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zip_iter_move{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    return _CUDA_VSTD::apply(__zip_iter_move{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  template <class _Tuple1, class _Tuple2, size_t... _Indices>
  _CCCL_API static constexpr void
  __iter_swap(_Tuple1&& __tuple1, _Tuple2&& __tuple2, _CUDA_VSTD::index_sequence<_Indices...>) noexcept(
    _CUDA_VSTD::__all<noexcept(_CUDA_VRANGES::iter_swap(
      _CUDA_VSTD::get<_Indices>(_CUDA_VSTD::declval<_Tuple1>()),
      _CUDA_VSTD::get<_Indices>(_CUDA_VSTD::declval<_Tuple2>())))...>::value)
  {
    (_CUDA_VRANGES::iter_swap(_CUDA_VSTD::get<_Indices>(_CUDA_VSTD::forward<_Tuple1>(__tuple1)),
                              _CUDA_VSTD::get<_Indices>(_CUDA_VSTD::forward<_Tuple2>(__tuple2))),
     ...);
  }

  template <class _Tuple1, class _Tuple2>
  _CCCL_API static constexpr void
  __iter_swap(_Tuple1&& __tuple1, _Tuple2&& __tuple2) noexcept(noexcept(__zip_iter_functors::__iter_swap(
    _CUDA_VSTD::forward<_Tuple1>(__tuple1),
    _CUDA_VSTD::forward<_Tuple2>(__tuple2),
    _CUDA_VSTD::make_index_sequence<_CUDA_VSTD::tuple_size<_CUDA_VSTD::remove_cvref_t<_Tuple1>>::value>())))
  {
    return __zip_iter_functors::__iter_swap(
      _CUDA_VSTD::forward<_Tuple1>(__tuple1),
      _CUDA_VSTD::forward<_Tuple2>(__tuple2),
      _CUDA_VSTD::make_index_sequence<_CUDA_VSTD::tuple_size<_CUDA_VSTD::remove_cvref_t<_Tuple1>>::value>());
  }
};

template <class... _Iterators>
struct __zip_iter_constraints
{
  static constexpr bool __all_forward       = (_CUDA_VSTD::forward_iterator<_Iterators> && ...);
  static constexpr bool __all_bidirectional = (_CUDA_VSTD::bidirectional_iterator<_Iterators> && ...);
  static constexpr bool __all_random_access = (_CUDA_VSTD::random_access_iterator<_Iterators> && ...);

  static constexpr bool __all_equality_comparable = (_CUDA_VSTD::equality_comparable<_Iterators> && ...);

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  static constexpr bool __all_three_way_comparable = (_CUDA_VSTD::three_way_comparable<_Iterators> && ...);
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  static constexpr bool __all_sized_sentinel = (_CUDA_VSTD::sized_sentinel_for<_Iterators, _Iterators> && ...);
  static constexpr bool __all_nothrow_iter_movable =
    (noexcept(_CUDA_VRANGES::iter_move(_CUDA_VSTD::declval<const _Iterators&>())) && ...)
    && (_CUDA_VSTD::is_nothrow_move_constructible_v<_CUDA_VSTD::iter_rvalue_reference_t<_Iterators>> && ...);

  static constexpr bool __all_indirectly_swappable = (_CUDA_VSTD::indirectly_swappable<_Iterators> && ...);

  static constexpr bool __all_noexcept_swappable = (_CUDA_VSTD::__noexcept_swappable<_Iterators> && ...);
};

struct __zv_iter_category_base_none
{};

struct __zv_iter_category_base_tag
{
  using iterator_category = _CUDA_VSTD::input_iterator_tag;
};

template <class... _Iterators>
using __zv_iter_category_base =
  _CUDA_VSTD::conditional_t<__zip_iter_constraints<_Iterators...>::__all_forward,
                            __zv_iter_category_base_tag,
                            __zv_iter_category_base_none>;

template <class... _Iterators>
_CCCL_API constexpr auto __get_zip_view_iterator_tag()
{
  using _Constraints = __zip_iter_constraints<_Iterators...>;
  if constexpr (_Constraints::__all_random_access)
  {
    return _CUDA_VSTD::random_access_iterator_tag();
  }
  else if constexpr (_Constraints::__all_bidirectional)
  {
    return _CUDA_VSTD::bidirectional_iterator_tag();
  }
  else if constexpr (_Constraints::__all_forward)
  {
    return _CUDA_VSTD::forward_iterator_tag();
  }
  else
  {
    return _CUDA_VSTD::input_iterator_tag();
  }
  _CCCL_UNREACHABLE();
}

template <class... _Iterators>
class zip_iterator : public __zv_iter_category_base<_Iterators...>
{
  __tuple_or_pair<_Iterators...> __current_;

  template <class...>
  friend class zip_iterator;

public:
  _CCCL_API constexpr explicit zip_iterator(__tuple_or_pair<_Iterators...> __iters)
      : __current_(_CUDA_VSTD::move(__iters))
  {}

  // We want to use the simpler `pair` if possible, but still want to be able to construct from a 3 element tuple
  _CCCL_TEMPLATE(size_t _NumIterators = sizeof...(_Iterators))
  _CCCL_REQUIRES((_NumIterators == 2))
  _CCCL_API constexpr explicit zip_iterator(_CUDA_VSTD::tuple<_Iterators...> __iters)
      : __current_(_CUDA_VSTD::get<0>(_CUDA_VSTD::move(__iters)), _CUDA_VSTD::get<1>(_CUDA_VSTD::move(__iters)))
  {}

  _CCCL_API constexpr explicit zip_iterator(_Iterators... __iters)
      : __current_(_CUDA_VSTD::move(__iters)...)
  {}

  using iterator_concept = decltype(__get_zip_view_iterator_tag<_Iterators...>());
  using value_type       = __tuple_or_pair<_CUDA_VSTD::iter_value_t<_Iterators>...>;
  using difference_type  = _CUDA_VSTD::common_type_t<_CUDA_VSTD::iter_difference_t<_Iterators>...>;

  _CCCL_HIDE_FROM_ABI zip_iterator() = default;

  template <class... _OtherIters>
  static constexpr bool __all_convertible =
    (_CUDA_VSTD::convertible_to<_OtherIters, _Iterators> && ...)
    && !(_CUDA_VSTD::is_same_v<_Iterators, _OtherIters> && ...);

  _CCCL_TEMPLATE(class... _OtherIters)
  _CCCL_REQUIRES((sizeof...(_OtherIters) == sizeof...(_Iterators)) _CCCL_AND __all_convertible<_OtherIters...>)
  _CCCL_API constexpr zip_iterator(zip_iterator<_OtherIters...> __i)
      : __current_(_CUDA_VSTD::move(__i.__current_))
  {}

  _CCCL_API constexpr auto operator*() const
  {
    return __zip_iter_functors::__iter_op_star(__current_);
  }

  _CCCL_API constexpr zip_iterator& operator++()
  {
    __zip_iter_functors::__iter_op_increment(__current_);
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

  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_bidirectional)
  _CCCL_API constexpr zip_iterator& operator--()
  {
    __zip_iter_functors::__iter_op_decrement(__current_);
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

  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_random_access)
  _CCCL_API constexpr zip_iterator& operator+=(difference_type __x)
  {
    __zip_iter_functors::__iter_op_pe(__x, __current_);
    return *this;
  }

  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_random_access)
  _CCCL_API constexpr zip_iterator& operator-=(difference_type __x)
  {
    __zip_iter_functors::__iter_op_me(__x, __current_);
    return *this;
  }

  _CCCL_TEMPLATE(class _Constraints = __zip_iter_constraints<_Iterators...>)
  _CCCL_REQUIRES(_Constraints::__all_random_access)
  _CCCL_API constexpr auto operator[](difference_type __n) const
  {
    return __zip_iter_functors::__iter_op_index(__n, __current_);
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator==(const zip_iterator& __x, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_equality_comparable)
  {
    if constexpr (_Constraints::__all_bidirectional)
    {
      return __x.__current_ == __y.__current_;
    }
    else
    {
      return __zip_iter_functors::__iter_op_eq(__x.__current_, __y.__current_);
    }
    _CCCL_UNREACHABLE();
  }

#if _CCCL_STD_VER <= 2017
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator!=(const zip_iterator& __x, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_equality_comparable)
  {
    if constexpr (_Constraints::__all_bidirectional)
    {
      return __x.__current_ != __y.__current_;
    }
    else
    {
      return !__zip_iter_functors::__iter_op_eq(__x.__current_, __y.__current_);
    }
    _CCCL_UNREACHABLE();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator<=>(const zip_iterator& __x, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access&& _Constraints::__all_three_way_comparable)
  {
    return __x.__current_ <=> __y.__current_;
  }

#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator<(const zip_iterator& __x, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return __x.__current_ < __y.__current_;
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator>(const zip_iterator& __x, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return __y < __x;
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator<=(const zip_iterator& __x, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return !(__y < __x);
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator>=(const zip_iterator& __x, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(bool)(_Constraints::__all_random_access)
  {
    return !(__x < __y);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator+(const zip_iterator& __i, difference_type __n)
    _CCCL_TRAILING_REQUIRES(zip_iterator)(_Constraints::__all_random_access)
  {
    auto __r = __i;
    __r += __n;
    return __r;
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator+(difference_type __n, const zip_iterator& __i)
    _CCCL_TRAILING_REQUIRES(zip_iterator)(_Constraints::__all_random_access)
  {
    return __i + __n;
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator-(const zip_iterator& __i, difference_type __n)
    _CCCL_TRAILING_REQUIRES(zip_iterator)(_Constraints::__all_random_access)
  {
    auto __r = __i;
    __r -= __n;
    return __r;
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto operator-(const zip_iterator& __x, const zip_iterator& __y)
    _CCCL_TRAILING_REQUIRES(difference_type)(_Constraints::__all_sized_sentinel)
  {
    return __zip_iter_functors::__iter_op_minus<difference_type>(__x.__current_, __y.__current_);
  }

  // MSVC falls over its feet if this is not a template
  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto iter_move(const zip_iterator& __i) noexcept(_Constraints::__all_nothrow_iter_movable)
  {
    return __zip_iter_functors::__iter_move(__i.__current_);
  }

  template <class _Constraints = __zip_iter_constraints<_Iterators...>>
  friend _CCCL_API constexpr auto
  iter_swap(const zip_iterator& __l, const zip_iterator& __r) noexcept(_Constraints::__all_noexcept_swappable)
    _CCCL_TRAILING_REQUIRES(void)(_Constraints::__all_indirectly_swappable)
  {
    return __zip_iter_functors::__iter_swap(__l.__current_, __r.__current_);
  }
};

template <class... _Iterators>
_CCCL_HOST_DEVICE zip_iterator(_CUDA_VSTD::tuple<_Iterators...>) -> zip_iterator<_Iterators...>;

template <class _Iterator1, class _Iterator2>
_CCCL_HOST_DEVICE zip_iterator(_CUDA_VSTD::pair<_Iterator1, _Iterator2>) -> zip_iterator<_Iterator1, _Iterator2>;

template <class... _Iterators>
_CCCL_HOST_DEVICE zip_iterator(_Iterators...) -> zip_iterator<_Iterators...>;

//! \p make_zip_iterator creates a \p zip_iterator from a \p tuple of iterators.
//!
//! \param t The \p tuple of iterators to copy.
//! \return A newly created \p zip_iterator which zips the iterators encapsulated in \p t.
template <typename... Iterators>
_CCCL_API constexpr zip_iterator<Iterators...> make_zip_iterator(_CUDA_VSTD::tuple<Iterators...> t)
{
  return zip_iterator<Iterators...>{_CUDA_VSTD::move(t)};
}

//! \p make_zip_iterator creates a \p zip_iterator from iterators.
//!
//! \param its The iterators to copy.
//! \return A newly created \p zip_iterator which zips the iterators.
template <typename... Iterators>
_CCCL_API constexpr zip_iterator<Iterators...> make_zip_iterator(Iterators... its)
{
  return zip_iterator<Iterators...>{_CUDA_VSTD::move(its)...};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

// GCC and MSVC2019 have issues determining _IsFancyPointer in C++17 because they fail to instantiate pointer_traits
#if (_CCCL_COMPILER(GCC) || _CCCL_COMPILER(MSVC)) && _CCCL_STD_VER <= 2017
_LIBCUDACXX_BEGIN_NAMESPACE_STD
template <class... _Iterators>
struct _IsFancyPointer<::cuda::zip_iterator<_Iterators...>> : false_type
{};
_LIBCUDACXX_END_NAMESPACE_STD
#endif // _CCCL_COMPILER(MSVC) && _CCCL_STD_VER <= 2017

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_ZIP_ITERATOR_H
