//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXECUTION_POLICY_H
#define _CUDA_STD___EXECUTION_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/underlying_type.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_EXECUTION

enum class __execution_policy : uint32_t
{
  __invalid_execution_policy = 0,
  __sequenced                = 1 << 0,
  __parallel                 = 1 << 1,
  __unsequenced              = 1 << 2,
  __parallel_unsequenced     = __execution_policy::__parallel | __execution_policy::__unsequenced,
};

[[nodiscard]] _CCCL_API constexpr bool
__satisfies_execution_policy(__execution_policy __lhs, __execution_policy __rhs) noexcept
{
  return (static_cast<uint32_t>(__lhs) & static_cast<uint32_t>(__rhs)) != 0;
}

template <__execution_policy _Policy>
struct __policy
{
  template <__execution_policy _OtherPolicy>
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __policy&, const __policy<_OtherPolicy>&) noexcept
  {
    using __underlying_t = underlying_type_t<__execution_policy>;
    return (static_cast<__underlying_t>(_Policy) == static_cast<__underlying_t>(_OtherPolicy));
  }

#if _CCCL_STD_VER <= 2017
  template <__execution_policy _OtherPolicy>
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __policy&, const __policy<_OtherPolicy>&) noexcept
  {
    using __underlying_t = underlying_type_t<__execution_policy>;
    return (static_cast<__underlying_t>(_Policy) != static_cast<__underlying_t>(_OtherPolicy));
  }
#endif // _CCCL_STD_VER <= 2017

  static constexpr __execution_policy __policy_ = _Policy;
};

struct sequenced_policy : public __policy<__execution_policy::__sequenced>
{};

_CCCL_GLOBAL_CONSTANT sequenced_policy seq{};

struct parallel_policy : public __policy<__execution_policy::__parallel>
{};
_CCCL_GLOBAL_CONSTANT parallel_policy par{};

struct parallel_unsequenced_policy : public __policy<__execution_policy::__parallel_unsequenced>
{};
_CCCL_GLOBAL_CONSTANT parallel_unsequenced_policy par_unseq{};

struct unsequenced_policy : public __policy<__execution_policy::__unsequenced>
{};
_CCCL_GLOBAL_CONSTANT unsequenced_policy unseq{};

_CCCL_END_NAMESPACE_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXECUTION_POLICY_H
