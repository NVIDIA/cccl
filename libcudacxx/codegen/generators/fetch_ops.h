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
#include <string>

#include "definitions.h"
#include <fmt/format.h>

static void FormatFetchOps(std::ostream& out)
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

  // Memory order dispatcher
  out << R"XXX(
template <class _Fn, class _Sco>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_memory_order_dispatch(_Fn& __cuda_fetch, int __memorder, _Sco) {
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: __cuda_atomic_fence(_Sco{}, __atomic_cuda_seq_cst{}); _CCCL_FALLTHROUGH();
        case __ATOMIC_CONSUME: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQUIRE: __cuda_fetch(__atomic_cuda_acquire{}); break;
        case __ATOMIC_ACQ_REL: __cuda_fetch(__atomic_cuda_acq_rel{}); break;
        case __ATOMIC_RELEASE: __cuda_fetch(__atomic_cuda_release{}); break;
        case __ATOMIC_RELAXED: __cuda_fetch(__atomic_cuda_relaxed{}); break;
        default: assert(0);
      }
    ),
    NV_IS_DEVICE, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQ_REL: __cuda_membar(_Sco{}); _CCCL_FALLTHROUGH();
        case __ATOMIC_CONSUME: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQUIRE: __cuda_fetch(__atomic_cuda_volatile{}); __cuda_membar(_Sco{}); break;
        case __ATOMIC_RELEASE: __cuda_membar(_Sco{}); __cuda_fetch(__atomic_cuda_volatile{}); break;
        case __ATOMIC_RELAXED: __cuda_fetch(__atomic_cuda_volatile{}); break;
        default: assert(0);
      }
    )
  )
}
)XXX";

  // Argument ID Reference
  // 0 - Atomic Operation
  // 1 - Operand Type
  // 2 - Operand Size
  // 3 - Type Constraint
  // 4 - Memory Order
  // 5 - Memory Order function tag
  // 6 - Scope Constraint
  // 7 - Scope function tag
  const std::string asm_intrinsic_format = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_{0}(
  _Type* __ptr, _Type& __dst, _Type __op, __atomic_cuda_{5}, __atomic_cuda_operand_{1}{2}, {7})
{{ asm volatile("atom.{0}{4}{6}.{1}{2} %0,[%1],%2;" : "={3}"(__dst) : "l"(__ptr), "{3}"(__op) : "memory"); }})XXX";

  // 0 - Atomic Operation
  // 1 - Operand type constraint
  const std::string fetch_bind_invoke = R"XXX(
template <typename _Type, typename _Tag, typename _Sco>
struct __cuda_atomic_bind_fetch_{0} {{
  _Type* __ptr;
  _Type* __dst;
  _Type* __op;

  template <typename _Atomic_Memorder>
  inline _CCCL_DEVICE void operator()(_Atomic_Memorder) {{
    __cuda_atomic_fetch_{0}(__ptr, *__dst, *__op, _Atomic_Memorder{{}}, _Tag{{}}, _Sco{{}});
  }}
}};
template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __cuda_atomic_fetch_{0}(_Type* __ptr, _Type& __dst, _Type __op, int __memorder, _Sco)
{{
  using __proxy_t        = typename __atomic_cuda_deduce_{1}<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_{1}<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  __cuda_atomic_bind_fetch_{0}<__proxy_t, __proxy_tag, _Sco> __bound_{0}{{__ptr_proxy, __dst_proxy, __op_proxy}};
  __cuda_atomic_fetch_memory_order_dispatch(__bound_{0}, __memorder, _Sco{{}});
}}
)XXX";

  constexpr size_t supported_sizes[] = {
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
            out << fmt::format(
              asm_intrinsic_format,
              /* 0 */ op_name,
              /* 1 */ operand(type),
              /* 2 */ size,
              /* 3 */ constraints(type, size),
              /* 4 */ semantic(sem),
              /* 5 */ semantic_tag(sem),
              /* 6 */ scope(sco),
              /* 7 */ scope_tag(sco));
          }
        }
      }
    }
    out << "\n" << fmt::format(fetch_bind_invoke, op_name, deduction);
  }
}

#endif // FETCH_OPS_H
