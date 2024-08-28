//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_UTILITY_H
#define __CUDAX_ASYNC_DETAIL_UTILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/initializer_list>

#include <cuda/experimental/__async/config.cuh>
#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/type_traits.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
_CCCL_GLOBAL_CONSTANT size_t _npos = ~0UL;

struct _ignore
{
  template <class... As>
  _CCCL_HOST_DEVICE constexpr _ignore(As&&...) noexcept {};
};

template <class...>
struct _undefined;

struct _empty
{};

struct [[deprecated]] _deprecated
{};

struct _nil
{};

struct _immovable
{
  _immovable() = default;
  _CUDAX_IMMOVABLE(_immovable);
};

_CCCL_HOST_DEVICE constexpr size_t _max(_CUDA_VSTD::initializer_list<size_t> il) noexcept
{
  size_t max = 0;
  for (auto i : il)
  {
    if (i > max)
    {
      max = i;
    }
  }
  return max;
}

_CCCL_HOST_DEVICE constexpr size_t _find_pos(bool const* const begin, bool const* const end) noexcept
{
  for (bool const* where = begin; where != end; ++where)
  {
    if (*where)
    {
      return static_cast<size_t>(where - begin);
    }
  }
  return _npos;
}

template <class Ty, class... Ts>
_CCCL_HOST_DEVICE constexpr size_t _index_of() noexcept
{
  constexpr bool _same[] = {_CUDA_VSTD::is_same_v<Ty, Ts>...};
  return __async::_find_pos(_same, _same + sizeof...(Ts));
}

template <class Ty, class Uy = Ty>
_CCCL_HOST_DEVICE constexpr Ty _exchange(Ty& obj, Uy&& new_value) noexcept
{
  constexpr bool _nothrow = //
    noexcept(Ty(static_cast<Ty&&>(obj))) && //
    noexcept(obj = static_cast<Uy&&>(new_value)); //
  static_assert(_nothrow);

  Ty old_value = static_cast<Ty&&>(obj);
  obj          = static_cast<Uy&&>(new_value);
  return old_value;
}

template <class Ty>
_CCCL_HOST_DEVICE constexpr void _swap(Ty& left, Ty& right) noexcept
{
  constexpr bool _nothrow = //
    noexcept(Ty(static_cast<Ty&&>(left))) && //
    noexcept(left = static_cast<Ty&&>(right)); //
  static_assert(_nothrow);

  Ty tmp = static_cast<Ty&&>(left);
  left   = static_cast<Ty&&>(right);
  right  = static_cast<Ty&&>(tmp);
}

template <class Ty>
_CCCL_HOST_DEVICE constexpr Ty _decay_copy(Ty&& ty) noexcept(_nothrow_decay_copyable<Ty>)
{
  return static_cast<Ty&&>(ty);
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wnon-template-friend")
_CCCL_NV_DIAG_SUPPRESS(probable_guiding_friend)

// _zip/_unzip is for keeping type names short. It has the unfortunate side
// effect of obfuscating the types.
namespace
{
template <size_t N>
struct _slot
{
  friend constexpr auto _slot_allocated(_slot<N>);
  static constexpr size_t value = N;
};

template <class Type, size_t N>
struct _allocate_slot : _slot<N>
{
  friend constexpr auto _slot_allocated(_slot<N>)
  {
    return static_cast<Type (*)()>(nullptr);
  }
};

template <class Type, size_t Id = 0, size_t Pow2 = 0>
constexpr size_t _next(long);

// If _slot_allocated(_slot<Id>) has NOT been defined, then SFINAE will keep this function out of the overload set...
template <class Type, //
          size_t Id   = 0,
          size_t Pow2 = 0,
          bool        = !_slot_allocated(_slot<Id + (1 << Pow2) - 1>())>
constexpr size_t _next(int)
{
  return __async::_next<Type, Id, Pow2 + 1>(0);
}

template <class Type, size_t Id, size_t Pow2>
constexpr size_t _next(long)
{
  if constexpr (Pow2 == 0)
  {
    return _allocate_slot<Type, Id>::value;
  }
  else
  {
    return __async::_next<Type, Id + (1 << (Pow2 - 1)), 0>(0);
  }
}

template <class Type, size_t Val = __async::_next<Type>(0)>
using _zip = _slot<Val>;

template <class Id>
using _unzip = decltype(_slot_allocated(Id())());

// burn the first slot
using _ignore_this_typedef [[maybe_unused]] = _zip<void>;
} // namespace

_CCCL_NV_DIAG_DEFAULT(probable_guiding_friend)
_CCCL_DIAG_POP

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
