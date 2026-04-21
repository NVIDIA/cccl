// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANGES_DROP_WHILE_VIEW_H
#define _CUDA_STD___RANGES_DROP_WHILE_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/ranges_find_if_not.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__functional/bind_back.h>
#include <cuda/std/__functional/reference_wrapper.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__ranges/non_propagating_cache.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

#if _CCCL_HAS_CONCEPTS()
template <view _View, class _Pred>
  requires input_range<_View> && ::cuda::std::is_object_v<_Pred>
        && ::cuda::std::indirect_unary_predicate<const _Pred, iterator_t<_View>>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _View,
          class _Pred,
          ::cuda::std::enable_if_t<view<_View>, int>                                                           = 0,
          ::cuda::std::enable_if_t<input_range<_View>, int>                                                    = 0,
          ::cuda::std::enable_if_t<::cuda::std::is_object_v<_Pred>, int>                                       = 0,
          ::cuda::std::enable_if_t<::cuda::std::indirect_unary_predicate<const _Pred, iterator_t<_View>>, int> = 0>
#endif // !_CCCL_HAS_CONCEPTS()
class drop_while_view : public view_interface<drop_while_view<_View, _Pred>>
{
  _CCCL_NO_UNIQUE_ADDRESS _View __base_{};
  _CCCL_NO_UNIQUE_ADDRESS __movable_box<_Pred> __pred_{};

  static constexpr bool _use_cache = forward_range<_View>;
  using _cache_type _CCCL_NODEBUG =
    ::cuda::std::conditional_t<_use_cache, __non_propagating_cache<iterator_t<_View>>, __empty_cache>;
  _CCCL_NO_UNIQUE_ADDRESS _cache_type __cached_begin_{};

public:
#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI drop_while_view()
    requires ::cuda::std::default_initializable<_View> && ::cuda::std::default_initializable<_Pred>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _View2 = _View, class _Pred2 = _Pred)
  _CCCL_REQUIRES(::cuda::std::default_initializable<_View2> _CCCL_AND ::cuda::std::default_initializable<_Pred2>)
  _CCCL_API constexpr drop_while_view() noexcept(::cuda::std::is_nothrow_default_constructible_v<_View2>
                                                 && ::cuda::std::is_nothrow_default_constructible_v<_Pred2>)
      : view_interface<drop_while_view<_View, _Pred>>{}
  {}
#endif // !_CCCL_HAS_CONCEPTS()

  _CCCL_API constexpr explicit drop_while_view(_View __base, _Pred __pred)
      : __base_{::cuda::std::move(__base)}
      , __pred_{::cuda::std::in_place, ::cuda::std::move(__pred)}

  {}

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(::cuda::std::copy_constructible<_View2>)
  [[nodiscard]] _CCCL_API constexpr _View base() const& noexcept(::cuda::std::is_nothrow_copy_constructible_v<_View>)
  {
    return __base_;
  }

  [[nodiscard]] _CCCL_API constexpr _View base() && noexcept(::cuda::std::is_nothrow_move_constructible_v<_View>)
  {
    return ::cuda::std::move(__base_);
  }

  [[nodiscard]] _CCCL_API constexpr const _Pred& pred() const
  {
    return *__pred_;
  }

  [[nodiscard]] _CCCL_API constexpr auto begin()
  {
    _CCCL_ASSERT(__pred_.__has_value(),
                 "drop_while_view needs to have a valid predicate before calling begin() -- did a previous "
                 "assignment to this drop_while_view fail?");

    if constexpr (_use_cache)
    {
      if (!__cached_begin_.__has_value())
      {
        __cached_begin_.__emplace(::cuda::std::ranges::find_if_not(__base_, ::cuda::std::cref(*__pred_)));
      }
      return *__cached_begin_;
    }
    else
    {
      return ::cuda::std::ranges::find_if_not(__base_, ::cuda::std::cref(*__pred_));
    }
    _CCCL_UNREACHABLE();
  }

  [[nodiscard]] _CCCL_API constexpr auto end()
  {
    return ::cuda::std::ranges::end(__base_);
  }
};

template <class _View, class _Pred>
inline constexpr bool enable_borrowed_range<drop_while_view<_View, _Pred>> = enable_borrowed_range<_View>;

template <class _Range, class _Pred>
_CCCL_HOST_DEVICE drop_while_view(_Range&&, _Pred) -> drop_while_view<::cuda::std::ranges::views::all_t<_Range>, _Pred>;

_CCCL_END_NAMESPACE_CUDA_STD_VIEWS

_CCCL_BEGIN_NAMESPACE_CUDA_STD_VIEWS
_CCCL_BEGIN_NAMESPACE_CPO(__drop_while)

struct __fn
{
  template <class _Range, class _Pred>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range, _Pred&& __pred) const
    noexcept(noexcept(drop_while_view{::cuda::std::forward<_Range>(__range), ::cuda::std::forward<_Pred>(__pred)}))
      -> decltype(drop_while_view{::cuda::std::forward<_Range>(__range), ::cuda::std::forward<_Pred>(__pred)})
  {
    return drop_while_view{::cuda::std::forward<_Range>(__range), ::cuda::std::forward<_Pred>(__pred)};
  }

  _CCCL_TEMPLATE(class _Pred)
  _CCCL_REQUIRES(::cuda::std::constructible_from<::cuda::std::decay_t<_Pred>, _Pred>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Pred&& __pred) const
    noexcept(::cuda::std::is_nothrow_constructible_v<::cuda::std::decay_t<_Pred>, _Pred>)
  {
    return ::cuda::std::ranges::__pipeable{::cuda::std::__bind_back(*this, ::cuda::std::forward<_Pred>(__pred))};
  }
};

_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto drop_while = __drop_while::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_VIEWS

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_DROP_WHILE_VIEW_H
