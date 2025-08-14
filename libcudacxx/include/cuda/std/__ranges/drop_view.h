// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_DROP_VIEW_H
#define _LIBCUDACXX___RANGES_DROP_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__functional/bind_back.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__fwd/string_view.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/empty_view.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/iota_view.h>
#include <cuda/std/__ranges/non_propagating_cache.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/repeat_view.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/subrange.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/auto_cast.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/forward_like.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_HAS_CONCEPTS()
template <view _View>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _View, enable_if_t<view<_View>, int> = 0>
#endif // !_CCCL_HAS_CONCEPTS()
class drop_view : public view_interface<drop_view<_View>>
{
  // We cache begin() whenever _CUDA_VRANGES::next is not guaranteed O(1) to provide an
  // amortized O(1) begin() method. If this is an input_range, then we cannot cache
  // begin because begin is not equality preserving.
  // Note: drop_view<input-range>::begin() is still trivially amortized O(1) because
  // one can't call begin() on it more than once.
  static constexpr bool _UseCache = forward_range<_View> && !(random_access_range<_View> && sized_range<_View>);
  using _Cache                    = _If<_UseCache, __non_propagating_cache<iterator_t<_View>>, __empty_cache>;
  _CCCL_NO_UNIQUE_ADDRESS _Cache __cached_begin_ = _Cache();
  _View __base_                                  = _View();
  range_difference_t<_View> __count_             = 0;

public:
#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI drop_view()
    requires default_initializable<_View>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(default_initializable<_View2>)
  _CCCL_API constexpr drop_view() noexcept(is_nothrow_default_constructible_v<_View2>)
      : view_interface<drop_view<_View>>()
  {}
#endif // !_CCCL_HAS_CONCEPTS()

  _CCCL_API constexpr drop_view(_View __base,
                                range_difference_t<_View> __count) noexcept(is_nothrow_move_constructible_v<_View>)
      : view_interface<drop_view<_View>>()
      , __base_(_CUDA_VSTD::move(__base))
      , __count_(__count)
  {
    _CCCL_ASSERT(__count_ >= 0, "count must be greater than or equal to zero.");
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(copy_constructible<_View2>)
  _CCCL_API constexpr _View base() const& noexcept(is_nothrow_copy_constructible_v<_View2>)
  {
    return __base_;
  }

  _CCCL_API constexpr _View base() && noexcept(is_nothrow_move_constructible_v<_View>)
  {
    return _CUDA_VSTD::move(__base_);
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES((!(__simple_view<_View2> && random_access_range<const _View2> && sized_range<const _View2>) ))
  _CCCL_API constexpr auto begin()
  {
    if constexpr (random_access_range<_View2> && sized_range<_View2>)
    {
      const auto __dist = _CUDA_VSTD::min(_CUDA_VRANGES::distance(__base_), __count_);
      return _CUDA_VRANGES::begin(__base_) + __dist;
    }
    else if constexpr (_UseCache)
    {
      if (!__cached_begin_.__has_value())
      {
        __cached_begin_.__emplace(
          _CUDA_VRANGES::next(_CUDA_VRANGES::begin(__base_), __count_, _CUDA_VRANGES::end(__base_)));
      }
      return *__cached_begin_;
    }
    else
    {
      return _CUDA_VRANGES::next(_CUDA_VRANGES::begin(__base_), __count_, _CUDA_VRANGES::end(__base_));
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(random_access_range<const _View2>&& sized_range<const _View2>)
  _CCCL_API constexpr auto begin() const
  {
    const auto __dist = _CUDA_VSTD::min(_CUDA_VRANGES::distance(__base_), __count_);
    return _CUDA_VRANGES::begin(__base_) + __dist;
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES((!__simple_view<_View2>) )
  _CCCL_API constexpr auto end()
  {
    return _CUDA_VRANGES::end(__base_);
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(range<const _View2>)
  _CCCL_API constexpr auto end() const
  {
    return _CUDA_VRANGES::end(__base_);
  }

  template <class _Self = drop_view>
  _CCCL_API static constexpr auto __size(_Self& __self)
  {
    const auto __s = _CUDA_VRANGES::size(__self.__base_);
    const auto __c = static_cast<decltype(__s)>(__self.__count_);
    return __s < __c ? 0 : __s - __c;
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(sized_range<_View2>)
  _CCCL_API constexpr auto size()
  {
    return __size(*this);
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(sized_range<const _View2>)
  _CCCL_API constexpr auto size() const
  {
    return __size(*this);
  }
};

template <class _Range>
_CCCL_HOST_DEVICE drop_view(_Range&&, range_difference_t<_Range>) -> drop_view<_CUDA_VIEWS::all_t<_Range>>;

template <class _Tp>
inline constexpr bool enable_borrowed_range<drop_view<_Tp>> = enable_borrowed_range<_Tp>;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__drop)

template <class _Tp>
inline constexpr bool __is_empty_view = false;

template <class _Tp>
inline constexpr bool __is_empty_view<empty_view<_Tp>> = true;

template <class _Tp>
inline constexpr bool __is_passthrough_specialization = false;

template <class _Tp, size_t _Extent>
inline constexpr bool __is_passthrough_specialization<span<_Tp, _Extent>> = true;

template <class _CharT, class _Traits>
inline constexpr bool __is_passthrough_specialization<basic_string_view<_CharT, _Traits>> = true;

template <class _Np, class _Bound>
inline constexpr bool __is_passthrough_specialization<iota_view<_Np, _Bound>> = true;

template <class _Iter, class _Sent, _CUDA_VRANGES::subrange_kind _Kind>
inline constexpr bool __is_passthrough_specialization<_CUDA_VRANGES::subrange<_Iter, _Sent, _Kind>> =
  !_CUDA_VRANGES::subrange<_Iter, _Sent, _Kind>::_StoreSize;

template <class _Tp>
inline constexpr bool __is_subrange_specialization_with_store_size = false;

template <class _Iter, class _Sent, _CUDA_VRANGES::subrange_kind _Kind>
inline constexpr bool __is_subrange_specialization_with_store_size<_CUDA_VRANGES::subrange<_Iter, _Sent, _Kind>> =
  _CUDA_VRANGES::subrange<_Iter, _Sent, _Kind>::_StoreSize;

template <class _Tp>
struct __passthrough_type;

template <class _Tp, size_t _Extent>
struct __passthrough_type<span<_Tp, _Extent>>
{
  using type = span<_Tp>;
};

template <class _CharT, class _Traits>
struct __passthrough_type<basic_string_view<_CharT, _Traits>>
{
  using type = basic_string_view<_CharT, _Traits>;
};

template <class _Np, class _Bound>
struct __passthrough_type<iota_view<_Np, _Bound>>
{
  using type = iota_view<_Np, _Bound>;
};

template <class _Iter, class _Sent, _CUDA_VRANGES::subrange_kind _Kind>
struct __passthrough_type<_CUDA_VRANGES::subrange<_Iter, _Sent, _Kind>>
{
  using type = _CUDA_VRANGES::subrange<_Iter, _Sent, _Kind>;
};

template <class _Tp>
using __passthrough_type_t = typename __passthrough_type<_Tp>::type;

template <class _Range, class _Np>
_CCCL_CONCEPT __use_empty = _CCCL_REQUIRES_EXPR((_Range, _Np))(
  requires(convertible_to<_Np, range_difference_t<_Range>>), requires(__is_empty_view<remove_cvref_t<_Range>>));

template <class _Range, class _Np>
_CCCL_CONCEPT __use_passthrough = _CCCL_REQUIRES_EXPR((_Range, _Np))(
  requires(convertible_to<_Np, range_difference_t<_Range>>),
  requires(!__is_empty_view<remove_cvref_t<_Range>>),
  requires(random_access_range<remove_cvref_t<_Range>>),
  requires(sized_range<remove_cvref_t<_Range>>),
  requires(__is_passthrough_specialization<remove_cvref_t<_Range>>));

template <class _Range, class _Np>
_CCCL_CONCEPT __use_subrange = _CCCL_REQUIRES_EXPR((_Range, _Np))(
  requires(convertible_to<_Np, range_difference_t<_Range>>),
  requires(!__is_empty_view<remove_cvref_t<_Range>>),
  requires(random_access_range<remove_cvref_t<_Range>>),
  requires(sized_range<remove_cvref_t<_Range>>),
  requires(__is_subrange_specialization_with_store_size<remove_cvref_t<_Range>>));

template <class _Range, class _Np>
_CCCL_CONCEPT __use_generic = _CCCL_REQUIRES_EXPR((_Range, _Np))(
  requires(!__use_empty<_Range, _Np>),
  requires(!__is_repeat_specialization<remove_cvref_t<_Range>>),
  requires(!__use_passthrough<_Range, _Np>),
  requires(!__use_subrange<_Range, _Np>));

struct __fn
{
  // [range.drop.overview]: the `empty_view` case.
  _CCCL_TEMPLATE(class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>)
  _CCCL_REQUIRES(__use_empty<_Range, _Np>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range, _Np&&) const
    noexcept(noexcept(/**/ _LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Range>(__range)))) -> _RawRange
  {
    return /*-----------*/ _LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Range>(__range));
  }

  // [range.drop.overview]: the `span | basic_string_view | iota_view | subrange (StoreSize == false)` case.
  _CCCL_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _CCCL_REQUIRES(__use_passthrough<_Range, _Np>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __rng, _Np&& __n) const
    // Note: deliberately not forwarding `__rng` to guard against double moves.
    noexcept(noexcept(__passthrough_type_t<_RawRange>(
      _CUDA_VRANGES::begin(__rng)
        + _CUDA_VSTD::min<_Dist>(_CUDA_VRANGES::distance(__rng), _CUDA_VSTD::forward<_Np>(__n)),
      _CUDA_VRANGES::end(__rng)))) -> __passthrough_type_t<_RawRange>
  {
    return __passthrough_type_t<_RawRange>(
      _CUDA_VRANGES::begin(__rng)
        + _CUDA_VSTD::min<_Dist>(_CUDA_VRANGES::distance(__rng), _CUDA_VSTD::forward<_Np>(__n)),
      _CUDA_VRANGES::end(__rng));
  }

  // [range.drop.overview]: the `subrange (StoreSize == true)` case.
  _CCCL_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _CCCL_REQUIRES(__use_subrange<_Range, _Np>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __rng, _Np&& __n) const
    // Note: deliberately not forwarding `__rng` to guard against double moves.
    noexcept(noexcept(_RawRange(
      _CUDA_VRANGES::begin(__rng)
        + _CUDA_VSTD::min<_Dist>(_CUDA_VRANGES::distance(__rng), _CUDA_VSTD::forward<_Np>(__n)),
      _CUDA_VRANGES::end(__rng),
      _CUDA_VSTD::__to_unsigned_like(
        _CUDA_VRANGES::distance(__rng)
        - _CUDA_VSTD::min<_Dist>(_CUDA_VRANGES::distance(__rng), _CUDA_VSTD::forward<_Np>(__n)))))) -> _RawRange
  {
    // Introducing local variables avoids calculating `min` and `distance` twice (at the cost of diverging from the
    // expression used in the `noexcept` clause and the return statement).
    auto dist    = _CUDA_VRANGES::distance(__rng);
    auto clamped = _CUDA_VSTD::min<_Dist>(dist, _CUDA_VSTD::forward<_Np>(__n));
    return _RawRange(
      _CUDA_VRANGES::begin(__rng) + clamped, _CUDA_VRANGES::end(__rng), _CUDA_VSTD::__to_unsigned_like(dist - clamped));
  }

  // [range.take.overview]: the `repeat_view` "_RawRange models sized_range" case.
  _CCCL_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _CCCL_REQUIRES(convertible_to<_Np, range_difference_t<_Range>> _CCCL_AND
                   __is_repeat_specialization<_RawRange> _CCCL_AND sized_range<_RawRange>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range, _Np&& __n) const
    noexcept(noexcept(_CUDA_VIEWS::repeat(
      _CUDA_VSTD::forward_like<_Range>(*__range.__value_),
      _CUDA_VRANGES::distance(__range)
        - _CUDA_VSTD::min<_Dist>(_CUDA_VRANGES::distance(__range), _CUDA_VSTD::forward<_Np>(__n))))) -> _RawRange
  {
    return _CUDA_VIEWS::repeat(
      _CUDA_VSTD::forward_like<_Range>(*__range.__value_),
      _CUDA_VRANGES::distance(__range)
        - _CUDA_VSTD::min<_Dist>(_CUDA_VRANGES::distance(__range), _CUDA_VSTD::forward<_Np>(__n)));
  }

  // [range.take.overview]: the `repeat_view` "otherwise" case.
  _CCCL_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _CCCL_REQUIRES(convertible_to<_Np, range_difference_t<_Range>> _CCCL_AND
                   __is_repeat_specialization<_RawRange> _CCCL_AND(!sized_range<_RawRange>))
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range, _Np&&) const
    noexcept(noexcept(/**/ _LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Range>(__range)))) -> _RawRange
  {
    return /*-----------*/ _LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Range>(__range));
  }

  // [range.drop.overview]: the "otherwise" case.
  _CCCL_TEMPLATE(class _Range, class _Np)
  _CCCL_REQUIRES(__use_generic<_Range, _Np>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range, _Np&& __n) const
    noexcept(noexcept(drop_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Np>(__n))))
      -> decltype(drop_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Np>(__n)))
  {
    return drop_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Np>(__n));
  }

  _CCCL_TEMPLATE(class _Np)
  _CCCL_REQUIRES(constructible_from<decay_t<_Np>, _Np>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Np&& __n) const
    noexcept(is_nothrow_constructible_v<decay_t<_Np>, _Np>)
  {
    return __pipeable(_CUDA_VSTD::__bind_back(*this, _CUDA_VSTD::forward<_Np>(__n)));
  }
};

_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto drop = __drop::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_DROP_VIEW_H
