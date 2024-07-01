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

#include "config.cuh"
#include "meta.cuh"
#include "type_traits.cuh"

// This must be the last #include
#include "prologue.cuh"

namespace cuda::experimental::__async
{
template <size_t Idx, class Ty>
struct _box
{
  // Too many compiler bugs with [[no_unique_address]] to use it here.
  // E.g., https://github.com/llvm/llvm-project/issues/88077
  // _CCCL_NO_UNIQUE_ADDRESS
  Ty _value;
};

template <size_t Idx, class Ty>
_CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE constexpr auto _cget(_box<Idx, Ty> const& box) noexcept -> Ty const&
{
  return box._value;
}

template <class Idx, class... Ts>
struct _tupl;

template <size_t... Idx, class... Ts>
struct _tupl<_mindices<Idx...>, Ts...> : _box<Idx, Ts>...
{
  template <class Fn, class Self, class... Us>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE static auto apply(Fn&& fn, Self&& self, Us&&... us) //
    noexcept(
      noexcept(static_cast<Fn&&>(fn)(static_cast<Us&&>(us)..., static_cast<Self&&>(self)._box<Idx, Ts>::_value...)))
      -> decltype(static_cast<Fn&&>(fn)(static_cast<Us&&>(us)..., static_cast<Self&&>(self)._box<Idx, Ts>::_value...))
  {
    return static_cast<Fn&&>(fn)(static_cast<Us&&>(us)..., static_cast<Self&&>(self)._box<Idx, Ts>::_value...);
  }

  template <class Fn, class Self, class... Us>
  _CUDAX_ALWAYS_INLINE _CCCL_HOST_DEVICE static auto for_each(Fn&& fn, Self&& self, Us&&... us) //
    noexcept((_nothrow_callable<Fn, Us..., _copy_cvref_t<Self, Ts>>
              && ...)) -> _mif<(_callable<Fn, Us..., _copy_cvref_t<Self, Ts>> && ...)>
  {
    return (static_cast<Fn&&>(fn)(static_cast<Us&&>(us)..., static_cast<Self&&>(self)._box<Idx, Ts>::_value), ...);
  }
};

template <class... Ts>
_CCCL_HOST_DEVICE _tupl(Ts...) //
  -> _tupl<_mmake_indices<sizeof...(Ts)>, Ts...>;

template <class... Ts>
using _tuple = _tupl<_mmake_indices<sizeof...(Ts)>, Ts...>;

template <class Fn, class Tupl, class... Us>
using _apply_result_t = decltype(DECLVAL(Tupl).apply(DECLVAL(Fn), DECLVAL(Tupl), DECLVAL(Us)...));

template <class First, class Second>
struct _pair
{
  First first;
  Second second;
};

template <class... Ts>
using _decayed_tuple = _tuple<_decay_t<Ts>...>;
} // namespace cuda::experimental::__async

#include "epilogue.cuh"
