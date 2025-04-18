//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_TUPLE
#define __CUDAX_ASYNC_DETAIL_TUPLE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__async/sender/meta.cuh>
#include <cuda/experimental/__async/sender/type_traits.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
template <size_t _Idx, class _Ty>
struct __box
{
  // Too many compiler bugs with [[no_unique_address]] to use it here.
  // E.g., https://github.com/llvm/llvm-project/issues/88077
  // _CCCL_NO_UNIQUE_ADDRESS
  _Ty __value_;
};

template <size_t _Idx, class _Ty>
_CUDAX_TRIVIAL_API constexpr auto __cget(__box<_Idx, _Ty> const& __box) noexcept -> _Ty const&
{
  return __box.__value_;
}

template <class _Idx, class... _Ts>
struct __tupl;

template <size_t... _Idx, class... _Ts>
struct __tupl<_CUDA_VSTD::index_sequence<_Idx...>, _Ts...> : __box<_Idx, _Ts>...
{
  template <class _Fn, class _Self, class... _Us>
  _CUDAX_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) //
    noexcept(__nothrow_callable<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>...>)
      -> __call_result_t<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>...>
  {
    return static_cast<_Fn&&>(__fn)( //
      static_cast<_Us&&>(__us)...,
      static_cast<_Self&&>(__self).__box<_Idx, _Ts>::__value_...);
  }
};

template <auto _Value>
using __v = _CUDA_VSTD::integral_constant<decltype(_Value), _Value>;

// Unroll tuples up to some size
#define _CCCL_TUPLE_DEFINE_TPARAM(_Idx)  , class _CCCL_PP_CAT(_T, _Idx)
#define _CCCL_TUPLE_INDEX_SEQUENCE(_Idx) , _Idx
#define _CCCL_TUPLE_TPARAM(_Idx)         , _CCCL_PP_CAT(_T, _Idx)
#define _CCCL_TUPLE_DEFINE_ELEMENT(_Idx) _CCCL_PP_CAT(_T, _Idx) _CCCL_PP_CAT(__t, _Idx);
#define _CCCL_TUPLE_CVREF_TPARAM(_Idx)   , __copy_cvref_t<_Self, _CCCL_PP_CAT(_T, _Idx)>
#define _CCCL_TUPLE_ELEMENT(_Idx)        , static_cast<_Self&&>(__self)._CCCL_PP_CAT(__t, _Idx)
#define _CCCL_TUPLE_MBR_PTR(_Idx)        , __v<&__tupl::_CCCL_PP_CAT(__t, _Idx)>

#define _CCCL_DEFINE_TUPLE(_SizeSub1)                                                                                 \
  template <class _T0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_DEFINE_TPARAM, 1)>                                       \
  struct __tupl<_CUDA_VSTD::index_sequence<0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_INDEX_SEQUENCE, 1)>,              \
                _T0 _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_TPARAM, 1)>                                                \
  {                                                                                                                   \
    _T0 __t0;                                                                                                         \
    _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_DEFINE_ELEMENT, 1)                                                         \
                                                                                                                      \
    template <class _Fn, class _Self, class... _Us>                                                                   \
    _CUDAX_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) noexcept(                       \
      __nothrow_callable<_Fn,                                                                                         \
                         _Us... _CCCL_TUPLE_CVREF_TPARAM(0) _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_CVREF_TPARAM, 1)>) \
      -> __call_result_t<_Fn,                                                                                         \
                         _Us... _CCCL_TUPLE_CVREF_TPARAM(0) _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_CVREF_TPARAM, 1)>  \
    {                                                                                                                 \
      return static_cast<_Fn&&>(__fn)(                                                                                \
        static_cast<_Us&&>(__us)... _CCCL_TUPLE_ELEMENT(0) _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_ELEMENT, 1));       \
    }                                                                                                                 \
                                                                                                                      \
    template <size_t _Idx>                                                                                            \
    _CUDAX_API static constexpr auto __get_mbr_ptr() noexcept                                                         \
    {                                                                                                                 \
      using __result_t =                                                                                              \
        _CUDA_VSTD::__type_index_c<_Idx, __v<&__tupl::__t0> _CCCL_PP_REPEAT(_SizeSub1, _CCCL_TUPLE_MBR_PTR, 1)>;      \
      return __result_t::value;                                                                                       \
    }                                                                                                                 \
  }

_CCCL_DEFINE_TUPLE(0);
_CCCL_DEFINE_TUPLE(1);
_CCCL_DEFINE_TUPLE(2);
_CCCL_DEFINE_TUPLE(3);
_CCCL_DEFINE_TUPLE(4);
_CCCL_DEFINE_TUPLE(5);
_CCCL_DEFINE_TUPLE(6);
_CCCL_DEFINE_TUPLE(7);

#undef _CCCL_TUPLE_DEFINE_TPARAM
#undef _CCCL_TUPLE_INDEX_SEQUENCE
#undef _CCCL_TUPLE_TPARAM
#undef _CCCL_TUPLE_DEFINE_ELEMENT
#undef _CCCL_TUPLE_CVREF_TPARAM
#undef _CCCL_TUPLE_ELEMENT
#undef _CCCL_TUPLE_MBR_PTR

template <size_t _Idx, class _Tupl, auto _MbrPtr = _Tupl::template __get_mbr_ptr<_Idx>()>
_CUDAX_TRIVIAL_API constexpr auto __cget(_Tupl const& __tupl) noexcept -> decltype(auto)
{
  return __tupl.*_MbrPtr;
}

template <class... _Ts>
__tupl(_Ts...) //
  ->__tupl<_CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>, _Ts...>;

template <class _Fn, class _Tupl, class... _Us>
using __apply_result_t = decltype(declval<_Tupl>().__apply(declval<_Fn>(), declval<_Tupl>(), declval<_Us>()...));

#if _CCCL_COMPILER(MSVC)
template <class... _Ts>
struct __mk_tuple_
{
  using __indices_t = _CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>;
  using type        = __tupl<__indices_t, _Ts...>;
};

template <class... _Ts>
using __tuple = typename __mk_tuple_<_Ts...>::type;
#else
template <class... _Ts>
using __tuple = __tupl<_CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>, _Ts...>;
#endif

template <class... _Ts>
using __decayed_tuple = __tuple<__decay_t<_Ts>...>;

// A very simple pair type
template <class _First, class _Second>
struct __pair
{
  _First first;
  _Second second;
};

template <class _First, class _Second>
__pair(_First, _Second) -> __pair<_First, _Second>;

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
