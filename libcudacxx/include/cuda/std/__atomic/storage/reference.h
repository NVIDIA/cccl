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

#ifndef _LIBCUDACXX___ATOMIC_STORAGE_REF_H
#define _LIBCUDACXX___ATOMIC_STORAGE_REF_H

#include <cuda/std/detail/__config>

#include <cuda/std/type_traits>

#include <cuda/std/__atomic/storage/base.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Reference is compatible with __atomic_base_tag and uses default dispatch

template <typename _Tp>
struct __atomic_ref_storage {
  using __underlying_t = _Tp;
  static constexpr __atomic_tag __tag = __atomic_tag::__atomic_base_tag;

#if !defined(_CCCL_COMPILER_GCC) || (__GNUC__ >= 5)
  static_assert(is_trivially_copyable<_Tp>::value,
    "std::atomic_ref<Tp> requires that 'Tp' be a trivially copyable type");
#endif

  _Tp* __a_value;

  __atomic_ref_storage() = delete;

  _CCCL_HOST_DEVICE constexpr explicit inline
  __atomic_ref_storage(_Tp* value) noexcept
    : __a_value(value) {}

  _CCCL_HOST_DEVICE inline
  auto get() -> __underlying_t* {
    return __a_value;
  }
  _CCCL_HOST_DEVICE inline
  auto get() const -> __underlying_t* {
    return __a_value;
  }
  _CCCL_HOST_DEVICE inline
  auto get() volatile -> volatile __underlying_t* {
    return __a_value;
  }
  _CCCL_HOST_DEVICE inline
  auto get() const volatile -> volatile __underlying_t* {
    return __a_value;
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMIC_STORAGE_REF_H
