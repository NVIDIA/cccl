//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_EQUAL_TO_VALUE_H
#define _CUDA___FUNCTIONAL_EQUAL_TO_VALUE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

/// @brief `__equal_to_value` is a function object that checks if a value is equal to
/// a specified value. It takes a single template parameter `_T`, which represents the
/// type of the value to be compared.
/// @tparam _T The type of the value to be compared.
template <typename _T>
struct __equal_to_value
{
  _T value_;

  explicit constexpr __equal_to_value(const T& value) noexcept
      : value_(value)
  {}

  __host__ __device__ constexpr bool operator()(const T& lhs) const noexcept {
    return lhs == value_;
  }
};

/// @brief `__equal_to_one` is a function object that checks if a value is equal to one.
/// It inherits from `__equal_to_value` and initializes it with the value `1`.
/// @tparam _T The type of the value to be compared.
template <typename _T>
struct __equal_to_one : __equal_to_value<T>
{
  constexpr __equal_to_one() noexcept
      : __equal_to_value<T>(static_cast<T>(1))
  {}
};

/// @brief `__equal_to_default` is a function object that checks if a value is equal to
/// the default value of its type. It inherits from `__equal_to_value` and initializes
/// it with the default-constructed value of type `_T`.
/// @tparam _T The type of the value to be compared.
template <typename _T>
struct __equal_to_default : __equal_to_value<T>
{
  constexpr __equal_to_default() noexcept
      : __equal_to_value<T>(T{})
  {}
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_EQUAL_TO_VALUE_H
