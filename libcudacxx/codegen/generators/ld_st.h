//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef LD_ST_H
#define LD_ST_H

#include <string>

#include "definitions.h"
#include <fmt/format.h>

static void FormatLoad(std::ostream& out)
{
  // Argument ID Reference
  // 0 - Operand Type
  // 1 - Operand Size
  // 2 - Constraint
  const std::string asm_intrinsic_format = R"XXX(
template <class _Type, class _Mmio, class _Scope, class _Memorder>
static inline _CCCL_DEVICE void __cuda_atomic_load(
  _Type* __ptr, _Type& __dst, __atomic_operand_tag<__atomic_operand_type::_{0},{1}>, _Mmio, _Scope, _Memorder)
{{
  asm volatile("ld.%2%3%4.{0}{1} %0,[%1];"
                : "={2}"(__dst)
                : "l"(__ptr), "C"(_Mmio::value), "C"(_Scope::value), "C"(_Memorder::value)
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
    2,
    4,
    8,
  };

  for (auto size : supported_sizes)
  {
    for (auto type : supported_types)
    {
      if (size == 2 && type == Operand::Floating)
      {
        continue;
      }
      out << fmt::format(asm_intrinsic_format, operand(type), size * 8, constraints(type, size));
    }
  }
}

static void FormatStore(std::ostream& out)
{
  // Argument ID Reference
  // 2 - Operand Type
  // 3 - Operand Size
  // 4 - Constraint
  const std::string asm_intrinsic_format = R"XXX(
template <class _Type, class _Mmio, class _Scope, class _Memorder>
static inline _CCCL_DEVICE void __cuda_atomic_store(
  _Type* __ptr, _Type __val, __atomic_operand_tag<__atomic_operand_type::_{0},{1}>, _Mmio, _Scope, _Memorder)
{{
  asm volatile("st.%2%3%4.{0}{1} %0,[%1];"
                :: "l"(__ptr), "{2}"(__val), "C"(_Mmio::value), "C"(_Scope::value), "C"(_Memorder::value)
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
    2,
    4,
    8,
  };

  for (auto size : supported_sizes)
  {
    for (auto type : supported_types)
    {
      if (size == 2 && type == Operand::Floating)
      {
        continue;
      }
      out << fmt::format(asm_intrinsic_format, operand(type), size * 8, constraints(type, size));
    }
  }
}

#endif // LD_ST_H
