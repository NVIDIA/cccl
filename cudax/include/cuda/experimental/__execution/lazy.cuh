//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_LAZY
#define __CUDAX_EXECUTION_LAZY

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
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>

#include <new> // IWYU pragma: keep

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
/// @brief A lazy type that can be used to delay the construction of a type.
template <class _Ty>
struct __lazy
{
  _CCCL_API __lazy() noexcept {}

  _CCCL_API ~__lazy() {}

  template <class... _Ts>
  _CCCL_API auto construct(_Ts&&... __ts) noexcept(__nothrow_constructible<_Ty, _Ts...>) -> _Ty&
  {
    _Ty* __value_ = ::new (static_cast<void*>(::cuda::std::addressof(__value_))) _Ty{static_cast<_Ts&&>(__ts)...};
    return *::cuda::std::launder(__value_);
  }

  template <class _Fn, class... _Ts>
  _CCCL_API auto construct_from(_Fn&& __fn, _Ts&&... __ts) noexcept(__nothrow_callable<_Fn, _Ts...>) -> _Ty&
  {
    _Ty* __value_ = ::new (static_cast<void*>(::cuda::std::addressof(__value_)))
      _Ty{static_cast<_Fn&&>(__fn)(static_cast<_Ts&&>(__ts)...)};
    return *::cuda::std::launder(__value_);
  }

  _CCCL_API void destroy() noexcept
  {
    ::cuda::std::destroy_at(&__value_);
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
using __lazy_box _CCCL_NODEBUG_ALIAS = __lazy_box_<_Idx, sizeof(_Ty), alignof(_Ty)>;
} // namespace __detail

template <class _Idx, class... _Ts>
struct __lazy_tupl;

template <>
struct __lazy_tupl<::cuda::std::index_sequence<>>
{
  template <class _Fn, class _Self, class... _Us>
  _CCCL_NODEBUG_API static auto __apply(_Fn&& __fn, _Self&&, _Us&&... __us) //
    noexcept(__nothrow_callable<_Fn, _Us...>) -> __call_result_t<_Fn, _Us...>
  {
    return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)...);
  }
};

template <size_t... _Idx, class... _Ts>
struct __lazy_tupl<::cuda::std::index_sequence<_Idx...>, _Ts...> : __detail::__lazy_box<_Idx, _Ts>...
{
  template <size_t _Ny>
  using __at _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_index_c<_Ny, _Ts...>;

  _CCCL_NODEBUG_API __lazy_tupl() noexcept {}

  _CCCL_API ~__lazy_tupl()
  {
    ((__engaged_[_Idx] ? ::cuda::std::destroy_at(__get<_Idx, _Ts>()) : void(0)), ...);
  }

  template <size_t _Ny, class _Ty>
  _CCCL_NODEBUG_API _Ty* __get() noexcept
  {
    return reinterpret_cast<_Ty*>(this->__detail::__lazy_box<_Ny, _Ty>::__data_);
  }

  template <size_t _Ny, class... _Us>
  _CCCL_NODEBUG_API __at<_Ny>& __emplace(_Us&&... __us) //
    noexcept(__nothrow_constructible<__at<_Ny>, _Us...>)
  {
    using _Ty _CCCL_NODEBUG_ALIAS = __at<_Ny>;
    _Ty* __value_                 = ::new (static_cast<void*>(__get<_Ny, _Ty>())) _Ty{static_cast<_Us&&>(__us)...};
    __engaged_[_Ny]               = true;
    return *::cuda::std::launder(__value_);
  }

  template <class _Fn, class _Self, class... _Us>
  _CCCL_NODEBUG_API static auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us) //
    noexcept(__nothrow_callable<_Fn, _Us..., ::cuda::std::__copy_cvref_t<_Self, _Ts>...>)
      -> __call_result_t<_Fn, _Us..., ::cuda::std::__copy_cvref_t<_Self, _Ts>...>
  {
    return static_cast<_Fn&&>(
      __fn)(static_cast<_Us&&>(__us)...,
            static_cast<::cuda::std::__copy_cvref_t<_Self, _Ts>&&>(*__self.template __get<_Idx, _Ts>())...);
  }

  bool __engaged_[sizeof...(_Ts)] = {};
};

#if _CCCL_COMPILER(MSVC)
template <class... _Ts>
struct __mk_lazy_tuple_
{
  using __indices_t _CCCL_NODEBUG_ALIAS = ::cuda::std::make_index_sequence<sizeof...(_Ts)>;
  using type _CCCL_NODEBUG_ALIAS        = __lazy_tupl<__indices_t, _Ts...>;
};

template <class... _Ts>
using __lazy_tuple _CCCL_NODEBUG_ALIAS = typename __mk_lazy_tuple_<_Ts...>::type;
#else // ^^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
template <class... _Ts>
using __lazy_tuple _CCCL_NODEBUG_ALIAS = __lazy_tupl<::cuda::std::make_index_sequence<sizeof...(_Ts)>, _Ts...>;
#endif // !_CCCL_COMPILER(MSVC)

template <class... _Ts>
using __decayed_lazy_tuple _CCCL_NODEBUG_ALIAS = __lazy_tuple<decay_t<_Ts>...>;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_LAZY
