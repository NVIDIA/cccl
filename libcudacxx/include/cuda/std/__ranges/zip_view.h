// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_ZIP_VIEW_H
#define _CUDA_STD___RANGES_ZIP_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__iterator/zip_iterator.h>
#include <cuda/std/__algorithm/ranges_min.h>
#include <cuda/std/__algorithm/ranges_min_element.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/empty_view.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__tuple_dir/apply.h>
#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

_LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE

struct __zv_functors
{
  // view functors
  struct __zip_begin
  {
    template <class... _Types>
    [[nodiscard]] _CCCL_API constexpr tuple<invoke_result_t<decltype(::cuda::std::ranges::begin)&, _Types>...>
    operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(tuple<invoke_result_t<decltype(::cuda::std::ranges::begin)&, _Types>...>{
        ::cuda::std::invoke(::cuda::std::ranges::begin, ::cuda::std::forward<_Types>(__tuple_elements))...}))
    {
      return {::cuda::std::invoke(::cuda::std::ranges::begin, ::cuda::std::forward<_Types>(__tuple_elements))...};
    }
  };

  template <class _Tuple>
  [[nodiscard]] _CCCL_API static constexpr auto __view_begin(_Tuple&& __tuple) noexcept(
    noexcept(::cuda::std::apply(__zv_functors::__zip_begin{}, ::cuda::std::forward<_Tuple>(__tuple))))
  {
    return ::cuda::std::apply(__zv_functors::__zip_begin{}, ::cuda::std::forward<_Tuple>(__tuple));
  }

  struct __zip_end
  {
    template <class... _Types>
    [[nodiscard]] _CCCL_API constexpr tuple<invoke_result_t<decltype(::cuda::std::ranges::end)&, _Types>...>
    operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(tuple<invoke_result_t<decltype(::cuda::std::ranges::end)&, _Types>...>{
        ::cuda::std::invoke(::cuda::std::ranges::end, ::cuda::std::forward<_Types>(__tuple_elements))...}))
    {
      return {::cuda::std::invoke(::cuda::std::ranges::end, ::cuda::std::forward<_Types>(__tuple_elements))...};
    }
  };

  template <class _Tuple>
  [[nodiscard]] _CCCL_API static constexpr auto __view_end(_Tuple&& __tuple) noexcept(
    noexcept(::cuda::std::apply(__zv_functors::__zip_end{}, ::cuda::std::forward<_Tuple>(__tuple))))
  {
    return ::cuda::std::apply(__zv_functors::__zip_end{}, ::cuda::std::forward<_Tuple>(__tuple));
  }

  struct __zip_size
  {
    template <class... _Sizes>
    [[nodiscard]] _CCCL_API static constexpr auto __get_min(_Sizes&&... __tuple_sizes) noexcept
    {
      using __size_type = make_unsigned_t<common_type_t<_Sizes...>>;
      return (::cuda::std::ranges::min) ({static_cast<__size_type>(__tuple_sizes)...});
    }

    template <class... _Types>
    [[nodiscard]] _CCCL_API constexpr auto operator()(_Types&&... __tuple_elements) const
      noexcept(noexcept(__zip_size::__get_min(
        ::cuda::std::invoke(::cuda::std::ranges::size, ::cuda::std::forward<_Types>(__tuple_elements))...)))
    {
      return __zip_size::__get_min(
        ::cuda::std::invoke(::cuda::std::ranges::size, ::cuda::std::forward<_Types>(__tuple_elements))...);
    }
  };

  template <class _Tuple>
  [[nodiscard]] _CCCL_API static constexpr auto __view_size(_Tuple&& __tuple) noexcept(
    noexcept(::cuda::std::apply(__zv_functors::__zip_size{}, ::cuda::std::forward<_Tuple>(__tuple))))
  {
    return ::cuda::std::apply(__zv_functors::__zip_size{}, ::cuda::std::forward<_Tuple>(__tuple));
  }

  // iterator functions
  template <class _Tuple1, class _Tuple2, size_t... _Indices>
  [[nodiscard]] _CCCL_API static constexpr bool
  __iter_op_eq(const _Tuple1& __tuple1, const _Tuple2& __tuple2, index_sequence<_Indices...>) noexcept(
    noexcept(((::cuda::std::get<_Indices>(__tuple1) == ::cuda::std::get<_Indices>(__tuple2)) || ...)))
  {
    return ((::cuda::std::get<_Indices>(__tuple1) == ::cuda::std::get<_Indices>(__tuple2)) || ...);
  }

  template <class _Tuple1, class _Tuple2>
  [[nodiscard]] _CCCL_API static constexpr bool
  __iter_op_eq(const _Tuple1& __tuple1, const _Tuple2& __tuple2) noexcept(noexcept(
    __zv_functors::__iter_op_eq(__tuple1, __tuple2, make_index_sequence<tuple_size_v<remove_cvref_t<_Tuple1>>>{})))
  {
    return __zv_functors::__iter_op_eq(__tuple1, __tuple2, make_index_sequence<tuple_size_v<remove_cvref_t<_Tuple1>>>{});
  }

  struct __op_comp_abs
  {
    // abs in cstdlib is not constexpr
    template <class _Diff>
    [[nodiscard]] _CCCL_API static constexpr _Diff __abs(_Diff __t) noexcept(noexcept(__t < 0 ? -__t : __t))
    {
      return __t < 0 ? -__t : __t;
    }

    template <class _Diff>
    [[nodiscard]] _CCCL_API constexpr bool operator()(const _Diff& __x, const _Diff& __y) const
      noexcept(noexcept(__op_comp_abs::__abs(__x) < __op_comp_abs::__abs(__y)))
    {
      return __op_comp_abs::__abs(__x) < __op_comp_abs::__abs(__y);
    }
  };

  template <class _Tp, class _Up>
  static constexpr bool __is_nothrow_differentiable = noexcept(declval<const _Tp&>() - declval<const _Up&>());

  template <class _Diff, class _Tuple1, class _Tuple2, size_t _Zero, size_t... _Indices>
  [[nodiscard]] _CCCL_API static constexpr _Diff
  __iter_op_minus(const _Tuple1& __tuple1, const _Tuple2& __tuple2, index_sequence<_Zero, _Indices...>) noexcept(
    __is_nothrow_differentiable<tuple_element_t<_Zero, _Tuple1>, tuple_element_t<_Zero, _Tuple2>>
    && (__is_nothrow_differentiable<tuple_element_t<_Indices, _Tuple1>, tuple_element_t<_Indices, _Tuple2>> && ...))
  {
    const _Diff __first = ::cuda::std::get<0>(__tuple1) - ::cuda::std::get<0>(__tuple2);
    if (__first == 0)
    {
      return __first;
    }

    const _Diff __temp[] = {__first, ::cuda::std::get<_Indices>(__tuple1) - ::cuda::std::get<_Indices>(__tuple2)...};
    return *(::cuda::std::ranges::min_element) (__temp, __op_comp_abs{});
  }

  template <class _Diff, class _Tuple1, class _Tuple2>
  [[nodiscard]] _CCCL_API static constexpr _Diff
  __iter_op_minus(const _Tuple1& __tuple1, const _Tuple2& __tuple2) noexcept(
    noexcept(__zv_functors::__iter_op_minus<_Diff>(
      __tuple1, __tuple2, make_index_sequence<tuple_size_v<remove_cvref_t<_Tuple1>>>{})))
  {
    return __zv_functors::__iter_op_minus<_Diff>(
      __tuple1, __tuple2, make_index_sequence<tuple_size_v<remove_cvref_t<_Tuple1>>>{});
  }
};

template <class... _Ranges>
_CCCL_CONCEPT __zip_is_common =
  (sizeof...(_Ranges) == 1 && (common_range<_Ranges> && ...))
  || (!(bidirectional_range<_Ranges> && ...) && (common_range<_Ranges> && ...))
  || ((random_access_range<_Ranges> && ...) && (sized_range<_Ranges> && ...));

template <bool _Const, class... _Views>
_CCCL_CONCEPT __zip_all_random_access = (random_access_range<__maybe_const<_Const, _Views>> && ...);

template <class... _Views>
_CCCL_CONCEPT __zip_all_input = (input_range<_Views> && ...);

template <class... _Views>
_CCCL_CONCEPT __zip_all_views = (view<_Views> && ...);

template <class... _Views>
struct __packed_views
{
  static constexpr bool __none_simple           = !(__simple_view<_Views> && ...);
  static constexpr bool __all_const_range       = (range<const _Views> && ...);
  static constexpr bool __all_sized_range       = (sized_range<_Views> && ...);
  static constexpr bool __all_const_sized_range = (sized_range<const _Views> && ...);
  static constexpr bool __not_empty             = sizeof...(_Views) > 0;
};

template <bool, class...>
class __zip_sentinel;

#if _CCCL_HAS_CONCEPTS()
template <input_range... _Views>
  requires(view<_Views> && ...) && (sizeof...(_Views) > 0)
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class... _Views>
#endif // !_CCCL_HAS_CONCEPTS()
class zip_view : public view_interface<zip_view<_Views...>>
{
#if _CCCL_STD_VER <= 2017
  static_assert(__zip_all_input<_Views...>, "zip_view requires input_range's as input");
  static_assert(__zip_all_views<_Views...>, "zip_view requires view's as input");
  static_assert(sizeof...(_Views) > 0, "zip_view requires a nonzero number of input ranges");
#endif // _CCCL_STD_VER <= 2017

  tuple<_Views...> __views_;

  template <bool _OtherConst>
  using __iterator = ::cuda::zip_iterator<iterator_t<__maybe_const<_OtherConst, _Views>>...>;

  template <bool _OtherConst>
  using __sentinel = __zip_sentinel<_OtherConst, _Views...>;

public:
  [[nodiscard]] _CCCL_API constexpr const tuple<_Views...>& base() const noexcept
  {
    return __views_;
  }

  _CCCL_HIDE_FROM_ABI zip_view() = default;

  _CCCL_API constexpr explicit zip_view(_Views... __views)
      : __views_{::cuda::std::move(__views)...}
  {}

  _CCCL_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _CCCL_REQUIRES(_Packed::__none_simple)
  [[nodiscard]] _CCCL_API constexpr auto begin()
  {
    return __iterator</*const*/ false>{__zv_functors::__view_begin(__views_)};
  }

  _CCCL_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _CCCL_REQUIRES(_Packed::__all_const_range)
  [[nodiscard]] _CCCL_API constexpr auto begin() const
  {
    return __iterator</*const*/ true>{__zv_functors::__view_begin(__views_)};
  }

  _CCCL_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _CCCL_REQUIRES(_Packed::__none_simple)
  [[nodiscard]] _CCCL_API constexpr auto end()
  {
    if constexpr (!__zip_is_common<_Views...>)
    {
      return __sentinel</*const*/ false>{__zv_functors::__view_end(__views_)};
    }
    else if constexpr (__zip_all_random_access</*const*/ false, _Views...>)
    {
      // MSVC cannot deal with iter_difference_t here
      using difference_type = common_type_t<range_difference_t<_Views>...>;
      return begin() + static_cast<difference_type>(size());
    }
    else
    {
      return __iterator</*const*/ false>{__zv_functors::__view_end(__views_)};
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _CCCL_REQUIRES(_Packed::__all_const_range)
  [[nodiscard]] _CCCL_API constexpr auto end() const
  {
    if constexpr (!__zip_is_common<const _Views...>)
    {
      return __sentinel</*const*/ true>{__zv_functors::__view_end(__views_)};
    }
    else if constexpr (__zip_all_random_access</*const*/ true, _Views...>)
    {
      // MSVC cannot deal with iter_difference_t here
      using difference_type = common_type_t<range_difference_t<const _Views>...>;
      return begin() + static_cast<difference_type>(size());
    }
    else
    {
      return __iterator</*const*/ true>{__zv_functors::__view_end(__views_)};
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _CCCL_REQUIRES(_Packed::__all_sized_range)
  [[nodiscard]] _CCCL_API constexpr auto size() noexcept(noexcept(__zv_functors::__view_size(__views_)))
  {
    return __zv_functors::__view_size(__views_);
  }

  _CCCL_TEMPLATE(class _Packed = __packed_views<_Views...>)
  _CCCL_REQUIRES(_Packed::__all_const_sized_range)
  [[nodiscard]] _CCCL_API constexpr auto size() const noexcept(noexcept(__zv_functors::__view_size(__views_)))
  {
    return __zv_functors::__view_size(__views_);
  }
};

template <class... _Ranges>
_CCCL_HOST_DEVICE zip_view(_Ranges&&...) -> zip_view<::cuda::std::ranges::views::all_t<_Ranges>...>;

template <bool _Const, class... _Views>
class __zip_sentinel
{
  template <class... _Iters>
  using __iterator = ::cuda::zip_iterator<_Iters...>;

  template <bool, class...>
  friend class __zip_sentinel;

  tuple<sentinel_t<__maybe_const<_Const, _Views>>...> __end_;

public:
  _CCCL_HIDE_FROM_ABI __zip_sentinel() = default;

  _CCCL_API constexpr explicit __zip_sentinel(tuple<sentinel_t<__maybe_const<_Const, _Views>>...> __end)
      : __end_{::cuda::std::move(__end)}
  {}

  template <bool _OtherConst>
  static constexpr bool __all_convertible =
    (convertible_to<sentinel_t<_Views>, sentinel_t<__maybe_const<_OtherConst, _Views>>> && ...);

  _CCCL_TEMPLATE(bool _OtherConst = _Const)
  _CCCL_REQUIRES(_OtherConst _CCCL_AND __all_convertible<_OtherConst>)
  _CCCL_API constexpr __zip_sentinel(__zip_sentinel<!_OtherConst, _Views...> __i)
      : __end_{::cuda::std::move(__i.__end_)}
  {}

  template <class... _Iters>
  static constexpr bool __is_iterator = same_as<__iterator<_Iters...>, __iterator<iterator_t<_Views>...>>;

  template <class... _Iters>
  static constexpr bool __is_const_iterator = same_as<__iterator<_Iters...>, __iterator<iterator_t<const _Views>...>>;

  template <class... _Iters>
  static constexpr bool __is_compatible_iterator =
    // sizeof...() check is technically not required, but it is much faster for compilers to
    // compute instead of needing to instantiate the other types
    (sizeof...(_Iters) == sizeof...(_Views)) && (__is_const_iterator<_Iters...> || __is_iterator<_Iters...>);

  // tie-breaker in case const and non-const iterators are the same
  template <class... _Iters>
  static constexpr bool __use_const_iterator = __is_const_iterator<_Iters...> && !__is_iterator<_Iters...>;

  template <bool _OtherConst>
  static constexpr bool __all_sentinel =
    (sentinel_for<sentinel_t<__maybe_const<_Const, _Views>>, iterator_t<__maybe_const<_OtherConst, _Views>>> && ...);

  template <class... _Iters, bool _OtherConst = __use_const_iterator<_Iters...>>
  [[nodiscard]] friend _CCCL_API constexpr auto operator==(const __iterator<_Iters...>& __x, const __zip_sentinel& __y)
    _CCCL_TRAILING_REQUIRES(bool)(__is_compatible_iterator<_Iters...>&& __all_sentinel<_OtherConst>)
  {
    return __zv_functors::__iter_op_eq(__x.__iterators(), __y.__end_);
  }

#if _CCCL_STD_VER <= 2017
  template <class... _Iters, bool _OtherConst = __use_const_iterator<_Iters...>>
  [[nodiscard]] friend _CCCL_API constexpr auto operator==(const __zip_sentinel& __x, const __iterator<_Iters...>& __y)
    _CCCL_TRAILING_REQUIRES(bool)(__is_compatible_iterator<_Iters...>&& __all_sentinel<_OtherConst>)
  {
    return __y == __x;
  }

  template <class... _Iters, bool _OtherConst = __use_const_iterator<_Iters...>>
  [[nodiscard]] friend _CCCL_API constexpr auto operator!=(const __iterator<_Iters...>& __x, const __zip_sentinel& __y)
    _CCCL_TRAILING_REQUIRES(bool)(__is_compatible_iterator<_Iters...>&& __all_sentinel<_OtherConst>)
  {
    return !(__x == __y);
  }

  template <class... _Iters, bool _OtherConst = __use_const_iterator<_Iters...>>
  [[nodiscard]] friend _CCCL_API constexpr auto operator!=(const __zip_sentinel& __x, const __iterator<_Iters...>& __y)
    _CCCL_TRAILING_REQUIRES(bool)(__is_compatible_iterator<_Iters...>&& __all_sentinel<_OtherConst>)
  {
    return __y != __x;
  }
#endif // _CCCL_STD_VER <= 2017

  template <bool _OtherConst>
  static constexpr bool __all_sized_sentinel =
    (sized_sentinel_for<sentinel_t<__maybe_const<_Const, _Views>>, iterator_t<__maybe_const<_OtherConst, _Views>>>
     && ...);

  template <bool _OtherConst>
  using _OtherDiff _CCCL_NODEBUG = common_type_t<range_difference_t<__maybe_const<_OtherConst, _Views>>...>;

  template <class... _Iters, bool _OtherConst = __use_const_iterator<_Iters...>>
  [[nodiscard]] friend _CCCL_API constexpr auto operator-(const __iterator<_Iters...>& __x, const __zip_sentinel& __y)
    _CCCL_TRAILING_REQUIRES(_OtherDiff<_OtherConst>)(
      __is_compatible_iterator<_Iters...>&& __all_sized_sentinel<_OtherConst>)
  {
    // Cannot reuse zip_iterator::operator-() here because that requires that lhs and rhs are
    // the same type, which they may not be here. Tuple does not define an operator-().
    return __zv_functors::__iter_op_minus<_OtherDiff<_OtherConst>>(__x.__iterators(), __y.__end_);
  }

  template <class... _Iters, bool _OtherConst = __use_const_iterator<_Iters...>>
  [[nodiscard]] friend _CCCL_API constexpr auto operator-(const __zip_sentinel& __x, const __iterator<_Iters...>& __y)
    _CCCL_TRAILING_REQUIRES(_OtherDiff<_OtherConst>)(
      __is_compatible_iterator<_Iters...>&& __all_sized_sentinel<_OtherConst>)
  {
    // Cannot reuse zip_iterator::operator-() here because that requires that lhs and rhs are
    // the same type, which they may not be here. Tuple does not define an operator-().
    return -(__zv_functors::__iter_op_minus<_OtherDiff<_OtherConst>>(__y.__iterators(), __x.__end_));
  }
};

_LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(zip_view)

template <class... _Views>
inline constexpr bool enable_borrowed_range<zip_view<_Views...>> = (enable_borrowed_range<_Views> && ...);

template <class... _Views>
inline constexpr bool __has_dangling_iterator<zip_view<_Views...>> = (__has_dangling_iterator<_Views> || ...);

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

_CCCL_BEGIN_NAMESPACE_CUDA_STD_VIEWS
_CCCL_BEGIN_NAMESPACE_CPO(__zip)

struct __fn
{
  [[nodiscard]] _CCCL_API constexpr empty_view<tuple<>> operator()() const noexcept
  {
    return empty<tuple<>>;
  }

  template <class... _Ranges>
  static constexpr bool __all_input = (input_range<_Ranges> && ...);

  _CCCL_TEMPLATE(class... _Ranges)
  _CCCL_REQUIRES((sizeof...(_Ranges) > 0) _CCCL_AND __all_input<_Ranges...>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Ranges&&... __rs) const
    noexcept(noexcept(zip_view<all_t<_Ranges&&>...>{::cuda::std::forward<_Ranges>(__rs)...}))
      -> decltype(zip_view<all_t<_Ranges&&>...>(std::forward<_Ranges>(__rs)...))
  {
    return /*------*/ zip_view<all_t<_Ranges>...>{::cuda::std::forward<_Ranges>(__rs)...};
  }
};

_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto zip = __zip::__fn{};
} // namespace __cpo
_CCCL_END_NAMESPACE_CUDA_STD_VIEWS

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_ZIP_VIEW_H
