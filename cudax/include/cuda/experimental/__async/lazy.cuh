//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_LAZY
#define __CUDAX_ASYNC_DETAIL_LAZY

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <new> // IWYU pragma: keep

namespace cuda::experimental::__async
{
/// @brief A lazy type that can be used to delay the construction of a type.
template <class _Ty>
struct __lazy
{
  _CUDAX_API __lazy() noexcept {}

  _CUDAX_API ~__lazy() {}

  template <class... _Ts>
  _CUDAX_API _Ty& construct(_Ts&&... __ts) noexcept(__nothrow_constructible<_Ty, _Ts...>)
  {
    _Ty* __value_ = ::new (static_cast<void*>(_CUDA_VSTD::addressof(__value_))) _Ty{static_cast<_Ts&&>(__ts)...};
    return *_CUDA_VSTD::launder(__value_);
  }

  template <class _Fn, class... _Ts>
  _CUDAX_API _Ty& construct_from(_Fn&& __fn, _Ts&&... __ts) noexcept(__nothrow_callable<_Fn, _Ts...>)
  {
    _Ty* __value_ = ::new (static_cast<void*>(_CUDA_VSTD::addressof(__value_)))
      _Ty{static_cast<_Fn&&>(__fn)(static_cast<_Ts&&>(__ts)...)};
    return *_CUDA_VSTD::launder(__value_);
  }

  _CUDAX_API void destroy() noexcept
  {
    _CUDA_VSTD::destroy_at(&__value_);
  }

  union
  {
    _Ty __value_;
  };
};

namespace __detail
{
template <size_t _Idx, size_t _Size, size_t _Align>
struct __lazy_box_
{
  static_assert(_Size != 0);
  alignas(_Align) unsigned char __data_[_Size];
};

template <size_t _Idx, class _Ty>
using __lazy_box = __lazy_box_<_Idx, sizeof(_Ty), alignof(_Ty)>;
} // namespace __detail

template <class _Idx, class... _Ts>
struct __lazy_tupl;

template <>
struct __lazy_tupl<_CUDA_VSTD::index_sequence<>>
{
  template <class _Fn, class _Self, class... _Us>
  _CUDAX_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&&, _Us&&... __us) //
    noexcept(__nothrow_callable<_Fn, _Us...>) -> __call_result_t<_Fn, _Us...>
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)...);
  }
};

template <size_t... _Idx, class... _Ts>
struct __lazy_tupl<_CUDA_VSTD::index_sequence<_Idx...>, _Ts...> : __detail::__lazy_box<_Idx, _Ts>...
{
  template <size_t _Ny>
  using __at = _CUDA_VSTD::__type_index_c<_Ny, _Ts...>;

  _CUDAX_TRIVIAL_API __lazy_tupl() noexcept {}

  _CUDAX_API ~__lazy_tupl()
  {
    ((__engaged_[_Idx] ? _CUDA_VSTD::destroy_at(__get<_Idx, _Ts>()) : void(0)), ...);
  }

  template <size_t _Ny, class _Ty>
  _CUDAX_TRIVIAL_API _Ty* __get() noexcept
  {
    return reinterpret_cast<_Ty*>(this->__detail::__lazy_box<_Ny, _Ty>::__data_);
  }

  template <size_t _Ny, class... _Us>
  _CUDAX_TRIVIAL_API __at<_Ny>& __emplace(_Us&&... __us) //
    noexcept(__nothrow_constructible<__at<_Ny>, _Us...>)
  {
    using _Ty       = __at<_Ny>;
    _Ty* __value_   = ::new (static_cast<void*>(__get<_Ny, _Ty>())) _Ty{static_cast<_Us&&>(__us)...};
    __engaged_[_Ny] = true;
    return *_CUDA_VSTD::launder(__value_);
  }

  template <class _Fn, class _Self, class... _Us>
  _CUDAX_TRIVIAL_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) //
    noexcept(__nothrow_callable<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>...>)
      -> __call_result_t<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>...>
  {
    return static_cast<_Fn&&>(__fn)(
      static_cast<_Us&&>(__us)..., static_cast<__copy_cvref_t<_Self, _Ts>&&>(*__self.template __get<_Idx, _Ts>())...);
  }

  bool __engaged_[sizeof...(_Ts)] = {};
};

#if defined(_CCCL_COMPILER_MSVC)
template <class... _Ts>
struct __mk_lazy_tuple_
{
  using __indices_t = _CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>;
  using type        = __lazy_tupl<__indices_t, _Ts...>;
};

template <class... _Ts>
using __lazy_tuple = typename __mk_lazy_tuple_<_Ts...>::type;
#else
template <class... _Ts>
using __lazy_tuple = __lazy_tupl<_CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>, _Ts...>;
#endif

template <class... _Ts>
using __decayed_lazy_tuple = __lazy_tuple<__decay_t<_Ts>...>;

} // namespace cuda::experimental::__async

#endif
