//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef FETCH_OPS_H
#define FETCH_OPS_H

#include <string>

#include "definitions.h"
#include <fmt/format.h>

static void FormatFetchOps(std::ostream& out)
{
  // Argument ID Reference
  // 0 - Operand Type
  // 1 - Operand Size
  // 2 - Constraint
  const std::string asm_intrinsic_format = R"XXX(
template <class _Type, class _Op, class _Scope, class _Memorder>
static inline _CCCL_DEVICE void __cuda_atomic_fetch(
  _Type* __ptr, _Type& __dst, _Type __op, __atomic_operand_tag<__atomic_operand_type::_{0},{1}>, _Op, _Scope, _Memorder)
{{
  asm volatile("atom.%3%4%5.{0}{1} %0,[%1],%2;"
                : "={2}"(__dst)
                : "l"(__ptr), "{2}"(__op), "C"(_Op::value), "C"(_Scope::value), "C"(_Memorder::value)
                : "memory");
}}
)XXX";

  constexpr Operand supported_types[] = {
    Operand::Bit,
    Operand::Floating,
    Operand::Unsigned,
    Operand::Signed,
  };

  constexpr size_t supported_sizes[] = {
    4,
    8,
  };

  for (auto size : supported_sizes)
  {
    for (auto type : supported_types)
    {
      out << fmt::format(asm_intrinsic_format, operand(type), size * 8, constraints(type, size));
    }
  }
}

#endif // FETCH_OPS_H
