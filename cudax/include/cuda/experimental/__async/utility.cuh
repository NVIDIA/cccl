//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_UTILITY
#define __CUDAX_ASYNC_DETAIL_UTILITY

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

#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
_CCCL_GLOBAL_CONSTANT size_t __npos = static_cast<size_t>(-1);

struct __ignore
{
  template <class... _As>
  _CUDAX_API constexpr __ignore(_As&&...) noexcept {};
};

using _CUDA_VSTD::__undefined; // NOLINT: misc-unused-using-decls

struct __empty
{};

struct [[deprecated]] __deprecated
{};

struct __nil
{};

struct __immovable
{
  __immovable() = default;
  _CUDAX_IMMOVABLE(__immovable);
};

_CUDAX_API constexpr size_t __maximum(_CUDA_VSTD::initializer_list<size_t> __il) noexcept
{
  size_t __max = 0;
  for (auto i : __il)
  {
    if (i > __max)
    {
      __max = i;
    }
  }
  return __max;
}

_CUDAX_API constexpr size_t __find_pos(bool const* const __begin, bool const* const __end) noexcept
{
  for (bool const* __where = __begin; __where != __end; ++__where)
  {
    if (*__where)
    {
      return static_cast<size_t>(__where - __begin);
    }
  }
  return __npos;
}

template <class _Ty, class... _Ts>
_CUDAX_API constexpr size_t __index_of() noexcept
{
  constexpr bool __same[] = {_CUDA_VSTD::is_same_v<_Ty, _Ts>...};
  return __async::__find_pos(__same, __same + sizeof...(_Ts));
}

template <class _Ty, class _Uy = _Ty>
_CUDAX_API constexpr _Ty __exchange(_Ty& __obj, _Uy&& __new_value) noexcept
{
  constexpr bool __is_nothrow = //
    noexcept(_Ty(static_cast<_Ty&&>(__obj))) && //
    noexcept(__obj = static_cast<_Uy&&>(__new_value)); //
  static_assert(__is_nothrow);

  _Ty old_value = static_cast<_Ty&&>(__obj);
  __obj         = static_cast<_Uy&&>(__new_value);
  return old_value;
}

template <class _Ty>
_CUDAX_API constexpr void __swap(_Ty& __left, _Ty& __right) noexcept
{
  constexpr bool __is_nothrow = //
    noexcept(_Ty(static_cast<_Ty&&>(__left))) && //
    noexcept(__left = static_cast<_Ty&&>(__right)); //
  static_assert(__is_nothrow);

  _Ty __tmp = static_cast<_Ty&&>(__left);
  __left    = static_cast<_Ty&&>(__right);
  __right   = static_cast<_Ty&&>(__tmp);
}

template <class _Ty>
_CUDAX_API constexpr _Ty __decay_copy(_Ty&& __ty) noexcept(__nothrow_decay_copyable<_Ty>)
{
  return static_cast<_Ty&&>(__ty);
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wnon-template-friend")
_CCCL_DIAG_SUPPRESS_NVHPC(probable_guiding_friend)
_CCCL_NV_DIAG_SUPPRESS(probable_guiding_friend)

// __zip/__unzip is for keeping type names short. It has the unfortunate side
// effect of obfuscating the types.
namespace
{
template <size_t _Ny>
struct __slot
{
  friend constexpr auto __slot_allocated(__slot<_Ny>);
};

template <class _Type, size_t _Ny>
struct __allocate_slot
{
  static constexpr size_t __value = _Ny;

  friend constexpr auto __slot_allocated(__slot<_Ny>)
  {
    return static_cast<_Type (*)()>(nullptr);
  }
};

template <class _Type, size_t _Id = 0, size_t _Pow2 = 0>
constexpr size_t __next(long);

// If __slot_allocated(__slot<_Id>) has NOT been defined, then SFINAE will keep
// this function out of the overload set...
template <class _Type, //
          size_t _Id   = 0,
          size_t _Pow2 = 0,
          bool         = !__slot_allocated(__slot<_Id + (1 << _Pow2) - 1>())>
constexpr size_t __next(int)
{
  return __async::__next<_Type, _Id, _Pow2 + 1>(0);
}

template <class _Type, size_t _Id, size_t _Pow2>
constexpr size_t __next(long)
{
  if constexpr (_Pow2 == 0)
  {
    return __allocate_slot<_Type, _Id>::__value;
  }
  else
  {
    return __async::__next<_Type, _Id + (1 << (_Pow2 - 1)), 0>(0);
  }
}

// Prior to Clang 12, we can't use the __slot trick to erase long type names
// because of a compiler bug. We'll just use the original type name in that case.
#if defined(_CCCL_COMPILER_CLANG) && _CCCL_CLANG_VERSION < 120000

template <class _Type>
using __zip = _Type;

template <class _Id>
using __unzip = _Id;

#else

template <class _Type, size_t _Val = __async::__next<_Type>(0)>
using __zip = __slot<_Val>;

template <class _Id>
using __unzip = decltype(__slot_allocated(_Id())());

#endif

// burn the first slot
using __ignore_this_typedef [[maybe_unused]] = __zip<void>;
} // namespace

_CCCL_NV_DIAG_DEFAULT(probable_guiding_friend)
_CCCL_DIAG_POP

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
