//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda/std/__memory/addressof.h>

#include <new>

#include "config.cuh"
#include "meta.cuh"
#include "type_traits.cuh"

namespace cuda::experimental::__async
{
/// @brief A lazy type that can be used to delay the construction of a type.
template <class Ty>
struct _lazy
{
  _CCCL_HOST_DEVICE _lazy() noexcept {}

  _CCCL_HOST_DEVICE ~_lazy() {}

  template <class... Ts>
  _CCCL_HOST_DEVICE Ty& construct(Ts&&... ts) noexcept(_nothrow_constructible<Ty, Ts...>)
  {
    ::new (static_cast<void*>(_CUDA_VSTD::addressof(value))) Ty(static_cast<Ts&&>(ts)...);
    return value;
  }

  template <class Fn, class... Ts>
  _CCCL_HOST_DEVICE Ty& construct_from(Fn&& fn, Ts&&... ts) noexcept(_nothrow_callable<Fn, Ts...>)
  {
    ::new (static_cast<void*>(_CUDA_VSTD::addressof(value))) Ty(static_cast<Fn&&>(fn)(static_cast<Ts&&>(ts)...));
    return value;
  }

  _CCCL_HOST_DEVICE void destroy() noexcept
  {
    value.~Ty();
  }

  union
  {
    Ty value;
  };
};

namespace _detail
{
template <size_t Idx, size_t Size, size_t Align>
struct _lazy_box_
{
  static_assert(Size != 0);
  alignas(Align) unsigned char _data[Size];
};

template <size_t Idx, class Ty>
using _lazy_box = _lazy_box_<Idx, sizeof(Ty), alignof(Ty)>;
} // namespace _detail

template <class Idx, class... Ts>
struct _lazy_tupl;

template <>
struct _lazy_tupl<_mindices<>>
{
  template <class Fn, class Self, class... Us>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE static auto apply(Fn&& fn, Self&& self, Us&&... us) //
    noexcept(_nothrow_callable<Fn, Us...>) -> _call_result_t<Fn, Us...>
  {
    return static_cast<Fn&&>(fn)(static_cast<Us&&>(us)...);
  }
};

template <size_t... Idx, class... Ts>
struct _lazy_tupl<_mindices<Idx...>, Ts...> : _detail::_lazy_box<Idx, Ts>...
{
  template <size_t Ny>
  using _at = _m_at_c<Ny, Ts...>;

  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE _lazy_tupl() noexcept {}

  _CCCL_HOST_DEVICE ~_lazy_tupl()
  {
    // casting the destructor expression to void is necessary for MSVC in
    // /permissive- mode.
    ((_engaged[Idx] ? void((*_get<Idx, Ts>()).~Ts()) : void(0)), ...);
  }

  template <size_t Ny, class Ty>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE Ty* _get() noexcept
  {
    return reinterpret_cast<Ty*>(this->_detail::_lazy_box<Ny, Ty>::_data);
  }

  template <size_t Ny, class... Us>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void _emplace(Us&&... us) //
    noexcept(_nothrow_constructible<_at<Ny>, Us...>)
  {
    using Ty = _at<Ny>;
    ::new (static_cast<void*>(_get<Ny, Ty>())) Ty{static_cast<Us&&>(us)...};
    _engaged[Ny] = true;
  }

  template <class Fn, class Self, class... Us>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE static auto apply(Fn&& fn, Self&& self, Us&&... us) //
    noexcept(_nothrow_callable<Fn, Us..., _copy_cvref_t<Self, Ts>...>)
      -> _call_result_t<Fn, Us..., _copy_cvref_t<Self, Ts>...>
  {
    return static_cast<Fn&&>(
      fn)(static_cast<Us&&>(us)..., static_cast<_copy_cvref_t<Self, Ts>&&>(*self.template _get<Idx, Ts>())...);
  }

  bool _engaged[sizeof...(Ts)] = {};
};

template <class... Ts>
using _lazy_tuple = _lazy_tupl<_mmake_indices<sizeof...(Ts)>, Ts...>;

template <class... Ts>
using _decayed_lazy_tuple = _lazy_tuple<_decay_t<Ts>...>;

} // namespace cuda::experimental::__async
