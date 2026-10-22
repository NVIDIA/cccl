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

#include <array>
#include <format>
#include <string>

#include "definitions.h"

inline std::string fetch_op_transform(std::string fetch_op)
{
  if (fetch_op == "add")
  {
    return "\n  __op = __op * __atomic_ptr_skip_t<_Type>::__skip;";
  }
  return {};
}

inline void FormatFetchOps(std::ostream& out)
{
  const std::vector arithmetic_types = {
    Operand::Floating,
    Operand::Unsigned,
    Operand::Signed,
  };

  const std::vector minmax_types = {
    Operand::Unsigned,
    Operand::Signed,
  };

  const std::vector bitwise_types = {Operand::Bit};

  const std::map op_support_map{
    std::pair{std::string{"add"}, std::pair{arithmetic_types, std::string{"arithmetic"}}},
    std::pair{std::string{"min"}, std::pair{minmax_types, std::string{"minmax"}}},
    std::pair{std::string{"max"}, std::pair{minmax_types, std::string{"minmax"}}},
    std::pair{std::string{"or"}, std::pair{bitwise_types, std::string{"bitwise"}}},
    std::pair{std::string{"xor"}, std::pair{bitwise_types, std::string{"bitwise"}}},
    std::pair{std::string{"and"}, std::pair{bitwise_types, std::string{"bitwise"}}},
  };

  // Argument ID Reference
  // 0 - Atomic Operation
  // 1 - Operand Type
  // 2 - Operand Size
  // 3 - Type Constraint
  // 4 - Memory Order
  // 5 - Memory Order function tag
  // 6 - Scope Constraint
  // 7 - Scope function tag
  constexpr auto asm_intrinsic_format = R"XXX(
template <class _Type>
_CCCL_DEVICE_API void __cuda_atomic_fetch_{0}(
  __cuda_atomic_ptx_backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, {5} __order, __cuda_atomic_operand_{1}{2}, {7})
{{ ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {7}{{}}); asm volatile("atom.{0}{4}{6}.{1}{2} %0,[%1],%2;" : "={3}"(__dst) : "l"(__ptr), "{3}"(__op) : "memory"); }})XXX";
  constexpr size_t supported_sizes[]  = {
    32,
    64,
  };

  constexpr Semantic supported_semantics[] = {
    Semantic::Acquire,
    Semantic::Relaxed,
    Semantic::Release,
    Semantic::Acq_Rel,
    Semantic::Volatile,
  };

  constexpr Scope supported_scopes[] = {
    Scope::CTA,
    Scope::Cluster,
    Scope::GPU,
    Scope::System,
  };

  for (auto& op_kp : op_support_map)
  {
    const auto& op_name    = op_kp.first;
    const auto& op_type_kp = op_kp.second;
    const auto& type_list  = op_type_kp.first;
    const auto& deduction  = op_type_kp.second;
    for (auto type : type_list)
    {
      for (auto size : supported_sizes)
      {
        const std::string proxy_type = operand_proxy_type(type, size);
        for (auto sco : supported_scopes)
        {
          for (auto sem : supported_semantics)
          {
            // There is no atom.add.s64
            if (op_name == "add" && type == Operand::Signed && size == 64)
            {
              continue;
            }
            out << std::format(
              asm_intrinsic_format,
              /* 0 */ op_name,
              /* 1 */ operand(type),
              /* 2 */ size,
              /* 3 */ constraints(type, size),
              /* 4 */ semantic(sem),
              /* 5 */ ptx_semantic_tag(sem),
              /* 6 */ scope(sco),
              /* 7 */ scope_tag(sco));
          }
        }
      }
    }
    out << "\n";
  }
}

#endif // FETCH_OPS_H
