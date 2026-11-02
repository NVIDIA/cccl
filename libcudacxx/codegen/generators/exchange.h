//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef EXCHANGE_H
#define EXCHANGE_H

#include <format>
#include <string>

#include "definitions.h"

inline void FormatExchange(std::ostream& out)
{
  // Argument ID Reference
  // 0 - Operand Type
  // 1 - Operand Size
  // 2 - Type Constraint
  // 3 - Memory Order
  // 4 - Memory Order function tag
  // 5 - Scope Constraint
  // 6 - Scope function tag
  constexpr auto asm_intrinsic_format_128 = R"XXX(
template <class _Type>
_CCCL_DEVICE_API void __cuda_atomic_exchange(
  __cuda_atomic_ptx_backend, _Type* __ptr, __unv<_Type>& __old, __unv<_Type> __new, {4} __order, __cuda_atomic_operand_{0}{1}, {6})
{{
  ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {6}{{}});
  static_assert(__cccl_ptx_isa >= 840 && (sizeof(_Type) == 16), "128b exchange is not supported until PTX ISA version 840");
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_90, (),
    NV_ANY_TARGET, (__atomic_exchange_128b_unsupported_before_SM_90();)
  )
  asm volatile(R"YYY(
    {{
      .reg .b128 _d;
      .reg .b128 _v;
      mov.b128 _v, {{%3, %4}};
      atom.exch{3}{5}.b128 _d,[%2],_v;
      mov.b128 {{%0, %1}}, _d;
    }}
  )YYY" : "=l"(__old.__x),"=l"(__old.__y) : "l"(__ptr), "l"(__new.__x),"l"(__new.__y) : "memory");
}})XXX";
  constexpr auto asm_intrinsic_format     = R"XXX(
template <class _Type>
_CCCL_DEVICE_API void __cuda_atomic_exchange(
  __cuda_atomic_ptx_backend, _Type* __ptr, __unv<_Type>& __old, __unv<_Type> __new, {4} __order, __cuda_atomic_operand_{0}{1}, {6})
{{ ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {6}{{}}); asm volatile("atom.exch{3}{5}.{0}{1} %0,[%1],%2;" : "={2}"(__old) : "l"(__ptr), "{2}"(__new) : "memory"); }})XXX";

  constexpr Operand supported_types[] = {
    Operand::Bit,
  };

  constexpr size_t supported_sizes[] = {
    32,
    64,
    128,
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

  for (auto size : supported_sizes)
  {
    for (auto type : supported_types)
    {
      for (auto sem : supported_semantics)
      {
        for (auto sco : supported_scopes)
        {
          if (size == 2 && type != Operand::Bit)
          {
            continue;
          }
          if (size == 128 && type != Operand::Bit)
          {
            continue;
          }

          if (size == 128)
          {
            out << std::format(
              asm_intrinsic_format_128,
              operand(type),
              size,
              constraints(type, size),
              semantic(sem),
              ptx_semantic_tag(sem),
              scope(sco),
              scope_tag(sco));
          }
          else
          {
            out << std::format(
              asm_intrinsic_format,
              operand(type),
              size,
              constraints(type, size),
              semantic(sem),
              ptx_semantic_tag(sem),
              scope(sco),
              scope_tag(sco));
          }
        }
      }
    }
  }

  out << "\n";
}

#endif // EXCHANGE_H
