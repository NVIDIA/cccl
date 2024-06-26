//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef COMPARED_AND_SWAP_H
#define COMPARED_AND_SWAP_H

#include <string>

#include "definitions.h"
#include <fmt/format.h>

static void FormatCompareAndSwap(std::ostream& out)
{
  // Argument ID Reference
  // 0 - Operand Type
  // 1 - Operand Size
  // 2 - Constraint
  const std::string asm_intrinsic_format = R"XXX(
template <class _Type, class _Scope, class _Memorder>
static inline _CCCL_DEVICE void __cuda_atomic_compare_exchange(
  _Type* __ptr, _Type& __dst, _Type __cmp, _Type __op, __atomic_operand_tag<__atomic_operand_type::_{0},{1}>, _Scope, _Memorder)
{{
  asm volatile("atom.cas.%4%5.{0}{1} %0,[%1],%2,%3;"
                : "={2}"(__dst)
                : "l"(__ptr), "{2}"(__cmp), "{2}"(__op), "C"(_Scope::value), "C"(_Memorder::value)
                : "memory");
}}
)XXX";

  constexpr Operand supported_types[] = {
    Operand::Bit,
    Operand::Floating,
    Operand::Signed,
    Operand::Unsigned,
  };

  constexpr size_t supported_sizes[] = {
    2,
    4,
    8,
  };

  for (auto size : supported_sizes)
  {
    for (auto type : supported_types)
    {
      if (size == 2 && type != Operand::Bit)
      {
        continue;
      }
      out << fmt::format(asm_intrinsic_format, operand(type), size * 8, constraints(type, size));
    }
  }
}

#endif // COMPARED_AND_SWAP_H
