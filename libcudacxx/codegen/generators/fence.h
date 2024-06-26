//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef FENCE_H
#define FENCE_H

#include <string>

static void FormatFence(std::ostream& out)
{
  // Argument ID Reference
  // 0 - Operand Type
  // 1 - Operand Size
  // 2 - Constraint
  const std::string asm_intrinsic_format = R"XXX(
template <class _Scope>
static inline _CCCL_DEVICE void __cuda_atomic_membar(_Scope)
{{
  asm volatile("membar%0;" :: "C"(_Scope::value) : "memory");
}}
template <class _Scope, class _Memorder>
static inline _CCCL_DEVICE void __cuda_atomic_fence(_Scope, _Memorder)
{{
  asm volatile("fence%0%1;" :: "C"(_Memorder::value), "C"(_Scope::value) : "memory");
}}
)XXX";

  out << asm_intrinsic_format;
}

#endif // FENCE_H
