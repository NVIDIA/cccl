// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_STORAGE_BASE_H
#define _LIBCUDACXX___ATOMIC_STORAGE_BASE_H

#include <cuda/std/detail/__config>

#include <cuda/std/type_traits>

#include <cuda/std/__atomic/storage/common.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __atomic_base_tag {};

template <typename _Tp>
struct __atomic_storage {
  using __underlying_t = _Tp;
  using __tag_t = __atomic_base_tag;

#if !defined(_CCCL_COMPILER_GCC) || (__GNUC__ >= 5)
  static_assert(is_trivially_copyable<_Tp>::value,
    "std::atomic<Tp> requires that 'Tp' be a trivially copyable type");
#endif

  _ALIGNAS(sizeof(_Tp)) _Tp __a_value;

  _LIBCUDACXX_HOST_DEVICE
  __atomic_storage() noexcept
    : __a_value() {}
  _LIBCUDACXX_HOST_DEVICE constexpr explicit
  __atomic_storage(_Tp value) noexcept
    : __a_value(value) {}

  _LIBCUDACXX_HOST_DEVICE inline auto operator()() -> __underlying_t* {
    return &__a_value;
  }
  _LIBCUDACXX_HOST_DEVICE inline auto operator()() volatile -> volatile __underlying_t* {
    return &__a_value;
  }
  _LIBCUDACXX_HOST_DEVICE inline auto operator()() const -> const __underlying_t* {
    return &__a_value;
  }
  _LIBCUDACXX_HOST_DEVICE inline auto operator()() const volatile -> const volatile __underlying_t* {
    return &__a_value;
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMIC_STORAGE_BASE_H
