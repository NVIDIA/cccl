// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_ZIP_VIEW_H
#define _LIBCUDACXX___RANGES_ZIP_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/ranges_min.h>
#include <cuda/std/__algorithm/ranges_min_element.h>
#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/__compare/three_way_comparable.h>
#endif
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/empty_view.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/tuple>

_CCCL_PUSH_MACROS

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

template <class... _Ranges>
_LIBCUDACXX_CONCEPT __zip_is_common =
  (sizeof...(_Ranges) == 1 && (common_range<_Ranges> && ...))
  || (!(bidirectional_range<_Ranges> && ...) && (common_range<_Ranges> && ...))
  || ((random_access_range<_Ranges> && ...) && (sized_range<_Ranges> && ...));

template <class _Tp, class _Up>
_LIBCUDACXX_HIDE_FROM_ABI auto __tuple_or_pair_test() -> pair<_Tp, _Up>;

template <class... _Types>
_LIBCUDACXX_HIDE_FROM_ABI auto __tuple_or_pair_test()
  _LIBCUDACXX_TRAILING_REQUIRES(tuple<_Types...>)((sizeof...(_Types) != 2));

template <class... _Types>
using __tuple_or_pair = decltype(__tuple_or_pair_test<_Types...>());

struct __zv_functors
{
  // view functors
  struct __zip_begin
  {
    template <class... _Types>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __tuple_or_pair<invoke_result_t<decltype(_CUDA_VRANGES::begin)&, _Types>...>
    operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(__tuple_or_pair<invoke_result_t<decltype(_CUDA_VRANGES::begin)&, _Types>...>{
        _CUDA_VSTD::invoke(_CUDA_VRANGES::begin, _CUDA_VSTD::forward<_Types>(__tuple_elements))...}))
    {
      return {_CUDA_VSTD::invoke(_CUDA_VRANGES::begin, _CUDA_VSTD::forward<_Types>(__tuple_elements))...};
    }
  };

  template <class _Tuple>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __view_begin(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zv_functors::__zip_begin{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    return _CUDA_VSTD::apply(__zv_functors::__zip_begin{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  struct __zip_end
  {
    template <class... _Types>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __tuple_or_pair<invoke_result_t<decltype(_CUDA_VRANGES::end)&, _Types>...>
    operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(__tuple_or_pair<invoke_result_t<decltype(_CUDA_VRANGES::end)&, _Types>...>{
        _CUDA_VSTD::invoke(_CUDA_VRANGES::end, _CUDA_VSTD::forward<_Types>(__tuple_elements))...}))
    {
      return {_CUDA_VSTD::invoke(_CUDA_VRANGES::end, _CUDA_VSTD::forward<_Types>(__tuple_elements))...};
    }
  };

  template <class _Tuple>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __view_end(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zv_functors::__zip_end{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    return _CUDA_VSTD::apply(__zv_functors::__zip_end{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  struct __zip_size
  {
    template <class... _Sizes>
    _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __get_min(_Sizes&&... __tuple_sizes) noexcept
    {
      using __size_type = __make_unsigned_t<common_type_t<_Sizes...>>;
      return (_CUDA_VRANGES::min)({static_cast<__size_type>(__tuple_sizes)...});
    }

    template <class... _Types>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Types&&... __tuple_elements) const noexcept(noexcept(
      __zip_size::__get_min(_CUDA_VSTD::invoke(_CUDA_VRANGES::size, _CUDA_VSTD::forward<_Types>(__tuple_elements))...)))
    {
      return __zip_size::__get_min(
        _CUDA_VSTD::invoke(_CUDA_VRANGES::size, _CUDA_VSTD::forward<_Types>(__tuple_elements))...);
    }
  };

  template <class _Tuple>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __view_size(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zv_functors::__zip_size{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    return _CUDA_VSTD::apply(__zv_functors::__zip_size{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  // iterator functions
  template <class _Tuple1, class _Tuple2, size_t... _Indices>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool
  __iter_op_eq(const _Tuple1& __tuple1, const _Tuple2& __tuple2, index_sequence<_Indices...>) noexcept(
    noexcept(((_CUDA_VSTD::get<_Indices>(__tuple1) == _CUDA_VSTD::get<_Indices>(__tuple2)) || ...)))
  {
    return ((_CUDA_VSTD::get<_Indices>(__tuple1) == _CUDA_VSTD::get<_Indices>(__tuple2)) || ...);
  }

  template <class _Tuple1, class _Tuple2>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool
  __iter_op_eq(const _Tuple1& __tuple1, const _Tuple2& __tuple2) noexcept(noexcept(__zv_functors::__iter_op_eq(
    __tuple1, __tuple2, _CUDA_VSTD::make_index_sequence<tuple_size_v<remove_cvref_t<_Tuple1>>>())))
  {
    return __zv_functors::__iter_op_eq(
      __tuple1, __tuple2, _CUDA_VSTD::make_index_sequence<tuple_size_v<remove_cvref_t<_Tuple1>>>());
  }

  struct __zip_op_star
  {
    struct __op_star
    {
      template <class _Iter>
      _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator()(_Iter& __i) const noexcept(noexcept(*__i))
      {
        return *__i;
      }
    };

    template <class... _Types>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(__tuple_or_pair<invoke_result_t<__op_star&, _Types>...>{
        _CUDA_VSTD::invoke(__op_star{}, _CUDA_VSTD::forward<_Types>(__tuple_elements))...}))
    {
      return __tuple_or_pair<invoke_result_t<__op_star&, _Types>...>{
        _CUDA_VSTD::invoke(__op_star{}, _CUDA_VSTD::forward<_Types>(__tuple_elements))...};
    }
  };

  template <class _Tuple>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __iter_op_star(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zv_functors::__zip_op_star{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    return _CUDA_VSTD::apply(__zv_functors::__zip_op_star{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  struct __zip_op_increment
  {
    struct __op_increment
    {
      template <class _Iter>
      _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator()(_Iter& __i) const noexcept(noexcept(++__i))
      {
        ++__i;
      }
    };

    template <class... _Types>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept((_CUDA_VSTD::invoke(__op_increment{}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...)))
    {
      (_CUDA_VSTD::invoke(__op_increment{}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...);
    }
  };

  template <class _Tuple>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __iter_op_increment(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zv_functors::__zip_op_increment{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    _CUDA_VSTD::apply(__zv_functors::__zip_op_increment{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  struct __zip_op_decrement
  {
    struct __op_decrement
    {
      template <class _Iter>
      _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator()(_Iter& __i) const noexcept(noexcept(--__i))
      {
        --__i;
      }
    };

    template <class... _Types>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept((_CUDA_VSTD::invoke(__op_decrement{}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...)))
    {
      (_CUDA_VSTD::invoke(__op_decrement{}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...);
    }
  };

  template <class _Tuple>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __iter_op_decrement(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zv_functors::__zip_op_decrement{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    _CUDA_VSTD::apply(__zv_functors::__zip_op_decrement{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  template <class _Diff>
  struct __zip_op_pe
  {
    _Diff __x;

    struct __op_pe
    {
      _Diff __x;

      template <class _Iter>
      _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator()(_Iter& __i) const
        noexcept(noexcept(__i += iter_difference_t<_Iter>(__x)))
      {
        __i += iter_difference_t<_Iter>(__x);
      }
    };

    template <class... _Types>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept((_CUDA_VSTD::invoke(__op_pe{__x}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...)))
    {
      (_CUDA_VSTD::invoke(__op_pe{__x}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...);
    }
  };

  template <class _Diff, class _Tuple>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __iter_op_pe(_Diff __x, _Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zv_functors::__zip_op_pe<_Diff>{__x}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    _CUDA_VSTD::apply(__zv_functors::__zip_op_pe<_Diff>{__x}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  template <class _Diff>
  struct __zip_op_me
  {
    _Diff __x;

    struct __op_me
    {
      _Diff __x;

      template <class _Iter>
      _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator()(_Iter& __i) const
        noexcept(noexcept(__i -= iter_difference_t<_Iter>(__x)))
      {
        __i -= iter_difference_t<_Iter>(__x);
      }
    };

    template <class... _Types>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept((_CUDA_VSTD::invoke(__op_me{__x}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...)))
    {
      (_CUDA_VSTD::invoke(__op_me{__x}, _CUDA_VSTD::forward<_Types>(__tuple_elements)), ...);
    }
  };

  template <class _Diff, class _Tuple>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __iter_op_me(_Diff __x, _Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zv_functors::__zip_op_me<_Diff>{__x}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    _CUDA_VSTD::apply(__zv_functors::__zip_op_me<_Diff>{__x}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  template <class _Diff>
  struct __zip_op_index
  {
    _Diff __n;

    struct __op_index
    {
      _Diff __n;

      template <class _Iter>
      _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator()(_Iter& __i) const
        noexcept(noexcept(__i[iter_difference_t<_Iter>(__n)]))
      {
        return __i[iter_difference_t<_Iter>(__n)];
      }
    };

    template <class... _Types>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(__tuple_or_pair<invoke_result_t<__zip_op_index::__op_index&, _Types>...>{
        _CUDA_VSTD::invoke(__zip_op_index::__op_index{__n}, _CUDA_VSTD::forward<_Types>(__tuple_elements))...}))
    {
      return __tuple_or_pair<invoke_result_t<__zip_op_index::__op_index&, _Types>...>{
        _CUDA_VSTD::invoke(__zip_op_index::__op_index{__n}, _CUDA_VSTD::forward<_Types>(__tuple_elements))...};
    }
  };

  template <class _Diff, class _Tuple>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __iter_op_index(_Diff __n, _Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zv_functors::__zip_op_index<_Diff>{__n}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    return _CUDA_VSTD::apply(__zv_functors::__zip_op_index<_Diff>{__n}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  struct __op_comp_abs
  {
    // abs in cstdlib is not constexpr
    template <class _Diff>
    _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Diff __abs(_Diff __t) noexcept(noexcept(__t < 0 ? -__t : __t))
    {
      return __t < 0 ? -__t : __t;
    }

    template <class _Diff>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator()(const _Diff& __x, const _Diff& __y) const
      noexcept(noexcept(__op_comp_abs::__abs(__x) < __op_comp_abs::__abs(__y)))
    {
      return __op_comp_abs::__abs(__x) < __op_comp_abs::__abs(__y);
    }
  };

  template <class _Diff, class _Tuple1, class _Tuple2, size_t _Zero, size_t... _Indices>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Diff
  __iter_op_minus(const _Tuple1& __tuple1, const _Tuple2& __tuple2, index_sequence<_Zero, _Indices...>) noexcept(
    noexcept(((_CUDA_VSTD::get<_Indices>(__tuple1) - _CUDA_VSTD::get<_Indices>(__tuple2)) && ...)))
  {
    const _Diff __first = _CUDA_VSTD::get<0>(__tuple1) - _CUDA_VSTD::get<0>(__tuple2);
    if (__first == 0)
    {
      return __first;
    }

    const _Diff __temp[] = {__first, _CUDA_VSTD::get<_Indices>(__tuple1) - _CUDA_VSTD::get<_Indices>(__tuple2)...};
    return *(_CUDA_VRANGES::min_element)(__temp, __op_comp_abs{});
  }

  template <class _Diff, class _Tuple1, class _Tuple2>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Diff
  __iter_op_minus(const _Tuple1& __tuple1, const _Tuple2& __tuple2) noexcept(
    noexcept(__zv_functors::__iter_op_minus<_Diff>(
      __tuple1, __tuple2, _CUDA_VSTD::make_index_sequence<tuple_size_v<remove_cvref_t<_Tuple1>>>())))
  {
    return __zv_functors::__iter_op_minus<_Diff>(
      __tuple1, __tuple2, _CUDA_VSTD::make_index_sequence<tuple_size_v<remove_cvref_t<_Tuple1>>>());
  }

  struct __zip_iter_move
  {
    template <class... _Types>
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __tuple_or_pair<invoke_result_t<decltype(_CUDA_VRANGES::iter_move)&, _Types>...>
    operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(__tuple_or_pair<invoke_result_t<decltype(_CUDA_VRANGES::iter_move)&, _Types>...>{
        _CUDA_VSTD::invoke(_CUDA_VRANGES::iter_move, _CUDA_VSTD::forward<_Types>(__tuple_elements))...}))
    {
      return {_CUDA_VSTD::invoke(_CUDA_VRANGES::iter_move, _CUDA_VSTD::forward<_Types>(__tuple_elements))...};
    }
  };

  template <class _Tuple>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __iter_move(_Tuple&& __tuple) noexcept(
    noexcept(_CUDA_VSTD::apply(__zip_iter_move{}, _CUDA_VSTD::forward<_Tuple>(__tuple))))
  {
    return _CUDA_VSTD::apply(__zip_iter_move{}, _CUDA_VSTD::forward<_Tuple>(__tuple));
  }

  template <class _Tuple1, class _Tuple2, size_t... _Indices>
  static constexpr bool __all_noexcept_swappable =
    (__noexcept_swappable<tuple_element<_Indices, _Tuple1>, tuple_element<_Indices, _Tuple2>> && ...);

  template <class _Tuple1, class _Tuple2, size_t... _Indices>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void
  __iter_swap(_Tuple1&& __tuple1, _Tuple2&& __tuple2, index_sequence<_Indices...>)
#  if !defined(_CCCL_COMPILER_GCC)
    noexcept(__all_noexcept_swappable<_Tuple1, _Tuple2, _Indices...>)
#  endif // !_CCCL_COMPILER_GCC
  {
    (_CUDA_VRANGES::iter_swap(_CUDA_VSTD::get<_Indices>(_CUDA_VSTD::forward<_Tuple1>(__tuple1)),
                              _CUDA_VSTD::get<_Indices>(_CUDA_VSTD::forward<_Tuple2>(__tuple2))),
     ...);
  }

  template <class _Tuple1, class _Tuple2>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __iter_swap(_Tuple1&& __tuple1, _Tuple2&& __tuple2) noexcept(noexcept(
    __zv_functors::__iter_swap(_CUDA_VSTD::forward<_Tuple1>(__tuple1),
                               _CUDA_VSTD::forward<_Tuple2>(__tuple2),
                               _CUDA_VSTD::make_index_sequence<tuple_size<remove_cvref_t<_Tuple1>>::value>())))
  {
    return __zv_functors::__iter_swap(
      _CUDA_VSTD::forward<_Tuple1>(__tuple1),
      _CUDA_VSTD::forward<_Tuple2>(__tuple2),
      _CUDA_VSTD::make_index_sequence<tuple_size<remove_cvref_t<_Tuple1>>::value>());
  }
};

template <bool _Const, class... _Views>
_LIBCUDACXX_CONCEPT __zip_all_random_access = (random_access_range<__maybe_const<_Const, _Views>> && ...);

template <bool _Const, class... _Views>
_LIBCUDACXX_CONCEPT __zip_all_bidirectional = (bidirectional_range<__maybe_const<_Const, _Views>> && ...);

template <bool _Const, class... _Views>
_LIBCUDACXX_CONCEPT __zip_all_forward = (forward_range<__maybe_const<_Const, _Views>> && ...);

template <class... _Views>
_LIBCUDACXX_CONCEPT __zip_all_input = (input_range<_Views> && ...);

template <class... _Views>
_LIBCUDACXX_CONCEPT __zip_all_views = (view<_Views> && ...);

template <bool _Const, class... _Views>
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto __get_zip_view_iterator_tag()
{
  if constexpr (__zip_all_random_access<_Const, _Views...>)
  {
    return random_access_iterator_tag();
  }
  else if constexpr (__zip_all_bidirectional<_Const, _Views...>)
  {
    return bidirectional_iterator_tag();
  }
  else if constexpr (__zip_all_forward<_Const, _Views...>)
  {
    return forward_iterator_tag();
  }
  else
  {
    return input_iterator_tag();
  }
  _CCCL_UNREACHABLE();
}

struct __zv_iter_category_base_none
{};

struct __zv_iter_category_base_tag
{
  using iterator_category = input_iterator_tag;
};

template <bool _Const, class... _Views>
using __zv_iter_category_base =
  _If<__zip_all_forward<_Const, _Views...>, __zv_iter_category_base_tag, __zv_iter_category_base_none>;

template <class... _Views>
struct __packed_views
{
  static constexpr bool __none_simple           = !(__simple_view<_Views> && ...);
  static constexpr bool __all_const_range       = (range<const _Views> && ...);
  static constexpr bool __all_sized_range       = (sized_range<_Views> && ...);
  static constexpr bool __all_const_sized_range = (sized_range<const _Views> && ...);
  static constexpr bool __not_empty             = sizeof...(_Views) > 0;
};

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

template <bool, class...>
class __zip_iterator;

template <bool, class...>
class __zip_sentinel;

#  if _CCCL_STD_VER >= 2020
template <input_range... _Views>
  requires(view<_Views> && ...) && (sizeof...(_Views) > 0)
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class... _Views>
#  endif // _CCCL_STD_VER <= 2017
class zip_view : public view_interface<zip_view<_Views...>>
{
  _CCCL_NO_UNIQUE_ADDRESS tuple<_Views...> __views_;

#  if _CCCL_STD_VER <= 2017 && !defined(_LIBCUDACXX_CUDACC_BELOW_11_3)
  static_assert(__zip_all_input<_Views...>, "zip_view requires input_range's as input");
  static_assert(__zip_all_views<_Views...>, "zip_view requires view's as input");
  static_assert(sizeof...(_Views) > 0, "zip_view requires a nonzero number of input ranges");
#  endif // _CCCL_STD_VER <= 2017 && !_LIBCUDACXX_CUDACC_BELOW_11_3

  template <bool _OtherConst>
  using __iterator = __zip_iterator<_OtherConst, _Views...>;

  template <bool _OtherConst>
  using __sentinel = __zip_sentinel<_OtherConst, _Views...>;

public:
  _CCCL_HIDE_FROM_ABI zip_view() = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit zip_view(_Views... __views)
      : __views_(_CUDA_VSTD::move(__views)...)
  {}

  _LIBCUDACXX_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _LIBCUDACXX_REQUIRES(_Packed::__none_simple)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin()
  {
    return __iterator<false>(__zv_functors::__view_begin(__views_));
  }

  _LIBCUDACXX_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _LIBCUDACXX_REQUIRES(_Packed::__all_const_range)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin() const
  {
    return __iterator<true>(__zv_functors::__view_begin(__views_));
  }

  _LIBCUDACXX_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _LIBCUDACXX_REQUIRES(_Packed::__none_simple)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end()
  {
    if constexpr (!__zip_is_common<_Views...>)
    {
      return __sentinel<false>(__zv_functors::__view_end(__views_));
    }
    else if constexpr (__zip_all_random_access<false, _Views...>)
    {
      return begin() + iter_difference_t<__iterator<false>>(size());
    }
    else
    {
      return __iterator<false>(__zv_functors::__view_end(__views_));
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _LIBCUDACXX_REQUIRES(_Packed::__all_const_range)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end() const
  {
    if constexpr (!__zip_is_common<const _Views...>)
    {
      return __sentinel<true>(__zv_functors::__view_end(__views_));
    }
    else if constexpr (__zip_all_random_access<true, _Views...>)
    {
      return begin() + iter_difference_t<__iterator<true>>(size());
    }
    else
    {
      return __iterator<true>(__zv_functors::__view_end(__views_));
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _LIBCUDACXX_REQUIRES(_Packed::__all_sized_range)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto size()
  {
    return __zv_functors::__view_size(__views_);
  }

  _LIBCUDACXX_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _LIBCUDACXX_REQUIRES(_Packed::__all_const_sized_range)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto size() const
  {
    return __zv_functors::__view_size(__views_);
  }
};

template <class... _Ranges>
_CCCL_HOST_DEVICE zip_view(_Ranges&&...) -> zip_view<_CUDA_VIEWS::all_t<_Ranges>...>;

template <bool _Const, class... _Views>
class __zip_iterator : public __zv_iter_category_base<_Const, _Views...>
{
  __tuple_or_pair<iterator_t<__maybe_const<_Const, _Views>>...> __current_;

  template <bool, class...>
  friend class __zip_iterator;

  template <bool, class...>
  friend class __zip_sentinel;

public:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __zip_iterator(
    __tuple_or_pair<iterator_t<__maybe_const<_Const, _Views>>...> __current)
      : __current_(_CUDA_VSTD::move(__current))
  {}

  using iterator_concept = decltype(__get_zip_view_iterator_tag<_Const, _Views...>());
  using value_type       = __tuple_or_pair<range_value_t<__maybe_const<_Const, _Views>>...>;
  using difference_type  = common_type_t<range_difference_t<__maybe_const<_Const, _Views>>...>;

  _CCCL_HIDE_FROM_ABI __zip_iterator() = default;

  template <bool _OtherConst>
  static constexpr bool __all_convertible =
    (convertible_to<iterator_t<_Views>, iterator_t<__maybe_const<_OtherConst, _Views>>> && ...);

  _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
  _LIBCUDACXX_REQUIRES(_OtherConst _LIBCUDACXX_AND __all_convertible<_OtherConst>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __zip_iterator(__zip_iterator<!_OtherConst, _Views...> __i)
      : __current_(_CUDA_VSTD::move(__i.__current_))
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator*() const
  {
    return __zv_functors::__iter_op_star(__current_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __zip_iterator& operator++()
  {
    __zv_functors::__iter_op_increment(__current_);
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
  _LIBCUDACXX_REQUIRES((!__zip_all_forward<_OtherConst, _Views...>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator++(int)
  {
    ++*this;
  }

  _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
  _LIBCUDACXX_REQUIRES(__zip_all_forward<_OtherConst, _Views...>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __zip_iterator operator++(int)
  {
    auto __tmp = *this;
    ++*this;
    return __tmp;
  }

  _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
  _LIBCUDACXX_REQUIRES(__zip_all_bidirectional<_OtherConst, _Views...>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __zip_iterator& operator--()
  {
    __zv_functors::__iter_op_decrement(__current_);
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
  _LIBCUDACXX_REQUIRES(__zip_all_bidirectional<_OtherConst, _Views...>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __zip_iterator operator--(int)
  {
    auto __tmp = *this;
    --*this;
    return __tmp;
  }

  _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
  _LIBCUDACXX_REQUIRES(__zip_all_random_access<_OtherConst, _Views...>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __zip_iterator& operator+=(difference_type __x)
  {
    __zv_functors::__iter_op_pe(__x, __current_);
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
  _LIBCUDACXX_REQUIRES(__zip_all_random_access<_OtherConst, _Views...>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __zip_iterator& operator-=(difference_type __x)
  {
    __zv_functors::__iter_op_me(__x, __current_);
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
  _LIBCUDACXX_REQUIRES(__zip_all_random_access<_OtherConst, _Views...>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator[](difference_type __n) const
  {
    return __zv_functors::__iter_op_index(__n, __current_);
  }

  template <bool _OtherConst>
  static constexpr bool __all_equality_comparable =
    (equality_comparable<iterator_t<__maybe_const<_OtherConst, _Views>>> && ...);

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator==(const __zip_iterator& __x, const __zip_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__all_equality_comparable<_OtherConst>)
  {
    if constexpr (__zip_all_bidirectional<_Const, _Views...>)
    {
      return __x.__current_ == __y.__current_;
    }
    else
    {
      return __zv_functors::__iter_op_eq(__x.__current_, __y.__current_);
    }
    _CCCL_UNREACHABLE();
  }

#  if _CCCL_STD_VER <= 2017
  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator!=(const __zip_iterator& __x, const __zip_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__all_equality_comparable<_OtherConst>)
  {
    if constexpr (__zip_all_bidirectional<_Const, _Views...>)
    {
      return __x.__current_ != __y.__current_;
    }
    else
    {
      return !__zv_functors::__iter_op_eq(__x.__current_, __y.__current_);
    }
    _CCCL_UNREACHABLE();
  }
#  endif // _CCCL_STD_VER <= 2017

#  ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  template <bool _OtherConst>
  static constexpr bool __all_three_way_comparable =
    (three_way_comparable<iterator_t<__maybe_const<_OtherConst, _Views>>> && ...);

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator<=>(const __zip_iterator& __x, const __zip_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(
      __zip_all_random_access<_OtherConst, _Views...>&& __all_three_way_comparable<_OtherConst>)
  {
    return __x.__current_ <=> __y.__current_;
  }

#  else // ^^^ !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR ^^^ / vvv _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR vvv

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator<(const __zip_iterator& __x, const __zip_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__zip_all_random_access<_OtherConst, _Views...>)
  {
    return __x.__current_ < __y.__current_;
  }

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator>(const __zip_iterator& __x, const __zip_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__zip_all_random_access<_OtherConst, _Views...>)
  {
    return __y < __x;
  }

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator<=(const __zip_iterator& __x, const __zip_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__zip_all_random_access<_OtherConst, _Views...>)
  {
    return !(__y < __x);
  }

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator>=(const __zip_iterator& __x, const __zip_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__zip_all_random_access<_OtherConst, _Views...>)
  {
    return !(__x < __y);
  }
#  endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator+(const __zip_iterator& __i, difference_type __n)
    _LIBCUDACXX_TRAILING_REQUIRES(__zip_iterator)(__zip_all_random_access<_OtherConst, _Views...>)
  {
    auto __r = __i;
    __r += __n;
    return __r;
  }

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator+(difference_type __n, const __zip_iterator& __i)
    _LIBCUDACXX_TRAILING_REQUIRES(__zip_iterator)(__zip_all_random_access<_OtherConst, _Views...>)
  {
    return __i + __n;
  }

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator-(const __zip_iterator& __i, difference_type __n)
    _LIBCUDACXX_TRAILING_REQUIRES(__zip_iterator)(__zip_all_random_access<_OtherConst, _Views...>)
  {
    auto __r = __i;
    __r -= __n;
    return __r;
  }

  template <bool _OtherConst>
  static constexpr bool __all_sized_sentinel =
    (sized_sentinel_for<iterator_t<__maybe_const<_OtherConst, _Views>>, iterator_t<__maybe_const<_OtherConst, _Views>>>
     && ...);

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator-(const __zip_iterator& __x, const __zip_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(difference_type)(__all_sized_sentinel<_OtherConst>)
  {
    return __zv_functors::__iter_op_minus<difference_type>(__x.__current_, __y.__current_);
  }

  template <bool _OtherConst>
  static constexpr bool __all_nothrow_iter_movable =
    (noexcept(_CUDA_VRANGES::iter_move(declval<const iterator_t<__maybe_const<_OtherConst, _Views>>&>())) && ...)
    && (is_nothrow_move_constructible_v<range_rvalue_reference_t<__maybe_const<_OtherConst, _Views>>> && ...);

  // MSVC falls over its feet if this is not a template
  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  iter_move(const __zip_iterator& __i) noexcept(__all_nothrow_iter_movable<_Const2>)
  {
    return __zv_functors::__iter_move(__i.__current_);
  }

  template <bool _OtherConst>
  static constexpr bool __all_indirectly_swappable =
    (indirectly_swappable<iterator_t<__maybe_const<_OtherConst, _Views>>> && ...);

  template <bool _OtherConst>
  static constexpr bool __all_noexcept_swappable =
    (__noexcept_swappable<iterator_t<__maybe_const<_OtherConst, _Views>>> && ...);

  template <bool _OtherConst = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  iter_swap(const __zip_iterator& __l, const __zip_iterator& __r) noexcept(__all_noexcept_swappable<_OtherConst>)
    _LIBCUDACXX_TRAILING_REQUIRES(void)(__all_indirectly_swappable<_OtherConst>)
  {
    return __zv_functors::__iter_swap(__l.__current_, __r.__current_);
  }
};

template <bool _Const, class... _Views>
class __zip_sentinel
{
  template <bool _OtherConst>
  using __iterator = __zip_iterator<_OtherConst, _Views...>;

  template <bool, class...>
  friend class __zip_sentinel;

  __tuple_or_pair<sentinel_t<__maybe_const<_Const, _Views>>...> __end_;

  // hidden friend cannot access private member of iterator because they are friends of friends
  template <bool _OtherConst>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr decltype(auto) __iter_current(__iterator<_OtherConst> const& __it)
  {
    return (__it.__current_);
  }

public:
  _CCCL_HIDE_FROM_ABI __zip_sentinel() = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __zip_sentinel(
    __tuple_or_pair<sentinel_t<__maybe_const<_Const, _Views>>...> __end)
      : __end_(__end)
  {}

  template <bool _OtherConst>
  static constexpr bool __all_convertible =
    (convertible_to<sentinel_t<_Views>, sentinel_t<__maybe_const<_OtherConst, _Views>>> && ...);

  _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
  _LIBCUDACXX_REQUIRES(_OtherConst _LIBCUDACXX_AND __all_convertible<_OtherConst>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __zip_sentinel(__zip_sentinel<!_OtherConst, _Views...> __i)
      : __end_(_CUDA_VSTD::move(__i.__end_))
  {}

  template <bool _OtherConst>
  static constexpr bool __all_sentinel =
    (sentinel_for<sentinel_t<__maybe_const<_Const, _Views>>, iterator_t<__maybe_const<_OtherConst, _Views>>> && ...);

  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator==(const __iterator<_OtherConst>& __x, const __zip_sentinel& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__all_sentinel<_OtherConst>)
  {
    return __zv_functors::__iter_op_eq(__iter_current(__x), __y.__end_);
  }

#  if _CCCL_STD_VER <= 2017
  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator==(const __zip_sentinel& __x, const __iterator<_OtherConst>& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__all_sentinel<_OtherConst>)
  {
    return __zv_functors::__iter_op_eq(__iter_current(__y), __x.__end_);
  }

  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator!=(const __iterator<_OtherConst>& __x, const __zip_sentinel& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__all_sentinel<_OtherConst>)
  {
    return !__zv_functors::__iter_op_eq(__iter_current(__x), __y.__end_);
  }

  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator!=(const __zip_sentinel& __x, const __iterator<_OtherConst>& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(__all_sentinel<_OtherConst>)
  {
    return !__zv_functors::__iter_op_eq(__iter_current(__y), __x.__end_);
  }
#  endif // _CCCL_STD_VER <= 2017

  template <bool _OtherConst>
  static constexpr bool __all_sized_sentinel =
    (sized_sentinel_for<sentinel_t<__maybe_const<_Const, _Views>>, iterator_t<__maybe_const<_OtherConst, _Views>>>
     && ...);

  template <bool _OtherConst>
  using _OtherDiff = common_type_t<range_difference_t<__maybe_const<_OtherConst, _Views>>...>;

  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator-(const __iterator<_OtherConst>& __x, const __zip_sentinel& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(_OtherDiff<_OtherConst>)(__all_sized_sentinel<_OtherConst>)
  {
    return __zv_functors::__iter_op_minus<_OtherDiff<_OtherConst>>(__iter_current(__x), __y.__end_);
  }

  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator-(const __zip_sentinel& __x, const __iterator<_OtherConst>& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(_OtherDiff<_OtherConst>)(__all_sized_sentinel<_OtherConst>)
  {
    return -(__zv_functors::__iter_op_minus<_OtherDiff<_OtherConst>>(__iter_current(__y), __x.__end_));
  }
};

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template <class... _Views>
_CCCL_INLINE_VAR constexpr bool enable_borrowed_range<zip_view<_Views...>> = (enable_borrowed_range<_Views> && ...);

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__zip)

struct __fn
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()() const noexcept
  {
    return empty_view<tuple<>>{};
  }

  template <class... _Ranges>
  static constexpr bool __all_input = (input_range<_Ranges> && ...);

  _LIBCUDACXX_TEMPLATE(class... _Ranges)
  _LIBCUDACXX_REQUIRES((sizeof...(_Ranges) > 0) _LIBCUDACXX_AND __all_input<_Ranges...>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Ranges&&... __rs) const
    noexcept(noexcept(zip_view<all_t<_Ranges>...>(_CUDA_VSTD::forward<_Ranges>(__rs)...))) -> zip_view<all_t<_Ranges>...>
  {
    return zip_view<all_t<_Ranges>...>(_CUDA_VSTD::forward<_Ranges>(__rs)...);
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto zip = __zip::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_VIEWS

// GCC has issues determining _IsFancyPointer in C++17 because it fails to instantiate pointer_traits
#  if defined(_CCCL_COMPILER_GCC) && _CCCL_STD_VER <= 2017

_LIBCUDACXX_BEGIN_NAMESPACE_STD
template <bool _Const, class... _Views>
struct _IsFancyPointer<_CUDA_VRANGES::__zip_iterator<_Const, _Views...>> : false_type
{};
_LIBCUDACXX_END_NAMESPACE_STD

#  endif // defined(_CCCL_COMPILER_GCC) && _CCCL_STD_VER <= 2017

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

_CCCL_POP_MACROS

#endif // _LIBCUDACXX___RANGES_ZIP_VIEW_H
