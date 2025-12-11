//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXPERIMENTAL_UTILITY_MANUAL_LIFETIME
#define __CUDAX_EXPERIMENTAL_UTILITY_MANUAL_LIFETIME

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental
{
/// @brief A lazy type that can be used to delay the construction of a type.
template <class _Ty>
struct __manual_lifetime
{
  template <class... _Ts>
  _CCCL_API auto __construct(_Ts&&... __ts) noexcept(::cuda::std::is_nothrow_constructible_v<_Ty, _Ts...>) -> _Ty&
  {
    // Use placement new directly instead of construct_at so we can use braced-init-list
    // construction
    _Ty* __value_ptr = ::new (static_cast<void*>(__data_)) _Ty{static_cast<_Ts&&>(__ts)...};
    return *::cuda::std::launder(__value_ptr);
  }

  template <class _Fn, class... _Ts>
  _CCCL_API auto __construct_from(_Fn&& __fn, _Ts&&... __ts) noexcept(::cuda::std::__is_nothrow_callable_v<_Fn, _Ts...>)
    -> _Ty&
  {
    // Use placement new directly instead of construct_at so we can use braced-init-list
    // construction
    _Ty* __value_ptr = ::new (static_cast<void*>(__data_)) _Ty{static_cast<_Fn&&>(__fn)(static_cast<_Ts&&>(__ts)...)};
    return *::cuda::std::launder(__value_ptr);
  }

  _CCCL_API auto __get() noexcept -> _Ty*
  {
    return reinterpret_cast<_Ty*>(__data_);
  }

  _CCCL_API auto __get() const noexcept -> const _Ty*
  {
    return reinterpret_cast<const _Ty*>(__data_);
  }

  _CCCL_API void __destroy() noexcept
  {
    __get()->~_Ty();
  }

  alignas(_Ty)::cuda::std::byte __data_[sizeof(_Ty)];
};
} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXPERIMENTAL_UTILITY_MANUAL_LIFETIME
