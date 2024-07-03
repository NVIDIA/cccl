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

static void FormatExchange(std::ostream& out)
{
  out << R"XXX(
template <class _Fn, class _Sco>
static inline _CCCL_DEVICE bool __cuda_atomic_exchange_memory_order_dispatch(_Fn& __cuda_exch, int __memorder, _Sco) {
  bool __res = false;
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__stronger_order_cuda(__memorder)) {
        case __ATOMIC_SEQ_CST: __cuda_atomic_fence(_Sco{}, __atomic_cuda_seq_cst{}); _CCCL_FALLTHROUGH();
        case __ATOMIC_CONSUME: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQUIRE: __res = __cuda_exch(__atomic_cuda_acquire{}); break;
        case __ATOMIC_ACQ_REL: __res = __cuda_exch(__atomic_cuda_acq_rel{}); break;
        case __ATOMIC_RELEASE: __res = __cuda_exch(__atomic_cuda_release{}); break;
        case __ATOMIC_RELAXED: __res = __cuda_exch(__atomic_cuda_relaxed{}); break;
        default: assert(0);
      }
    ),
    NV_IS_DEVICE, (
      switch (__stronger_order_cuda(__memorder)) {
        case __ATOMIC_SEQ_CST: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQ_REL: __cuda_atomic_membar(_Sco{}); _CCCL_FALLTHROUGH();
        case __ATOMIC_CONSUME: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQUIRE: __res = __cuda_exch(__atomic_cuda_volatile{}); __cuda_atomic_membar(_Sco{}); break;
        case __ATOMIC_RELEASE: __cuda_atomic_membar(_Sco{}); __res = __cuda_exch(__atomic_cuda_volatile{}); break;
        case __ATOMIC_RELAXED: __res = __cuda_exch(__atomic_cuda_volatile{}); break;
        default: assert(0);
      }
    )
  )
  return __res;
}
)XXX";

  // Argument ID Reference
  // 0 - Operand Type
  // 1 - Operand Size
  // 2 - Type Constraint
  // 3 - Memory Order
  // 4 - Memory Order function tag
  // 5 - Scope Constraint
  // 6 - Scope function tag
  const std::string asm_intrinsic_format_128 = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE bool __cuda_atomic_exchange(
  _Type* __ptr, _Type& __dst, _Type __cmp, _Type __op, {4}, __atomic_cuda_operand_{0}{1}, {6})
{{
  asm volatile(R"YYY(
.reg .b128 _d;
.reg .b128 _v;
mov.b128 {{%0, %1}}, _d;
mov.b128 {{%4, %5}}, _v;
atom.exch{3}{5}.b128 _d,[%2],_d,_v;
mov.b128 _d, {{%0, %1}};
)YYY" : "=l"(__dst.x),"=l"(__dst.y) : "l"(__ptr), "l"(__cmp.x),"l"(__cmp.y), "l"(__op.x),"l"(__op.y) : "memory"); return __dst.x == __cmp.x && __dst.y == __cmp.y; }})XXX";

  const std::string asm_intrinsic_format = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE bool __cuda_atomic_exchange(
  _Type* __ptr, _Type& __dst, _Type __cmp, _Type __op, {4}, __atomic_cuda_operand_{0}{1}, {6})
{{ asm volatile("atom.exch{3}{5}.{0}{1} %0,[%1],%2,%3;" : "={2}"(__dst) : "l"(__ptr), "{2}"(__cmp), "{2}"(__op) : "memory"); return __dst == __cmp; }})XXX";

  constexpr Operand supported_types[] = {
    Operand::Bit,
  };

  constexpr size_t supported_sizes[] = {
    16,
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
          out << fmt::format(
            (size == 128) ? asm_intrinsic_format_128 : asm_intrinsic_format,
            operand(type),
            size,
            constraints(type, size),
            semantic(sem),
            semantic_tag(sem),
            scope(sco),
            scope_tag(sco));
        }
      }
    }
  }

  out << std::endl
      << R"XXX(
template <typename _Type, typename _Tag, typename _Sco>
struct __cuda_atomic_bind_compare_exchange {
  _Type* __ptr;
  _Type* __exp;
  _Type* __des;

  template <typename _Atomic_Memorder>
  inline _CCCL_DEVICE bool operator()(_Atomic_Memorder) {
    return __cuda_atomic_compare_exchange(__ptr, *__exp, *__exp, *__des, _Atomic_Memorder{}, _Tag{}, _Sco{});
  }
};
template <class _Type, class _Sco>
static inline _CCCL_DEVICE bool __atomic_compare_exchange_cuda(_Type* __ptr, _Type& __exp, _Type __des, int __memorder, int __failure_memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __exp_proxy = reinterpret_cast<__proxy_t*>(&__exp);
  __proxy_t* __des_proxy  = reinterpret_cast<__proxy_t*>(&__des);
  __cuda_atomic_bind_compare_exchange<__proxy_t, __proxy_tag, _Sco> __bound_compare_swap{__ptr_proxy, __exp_proxy, __des_proxy};
  __cuda_atomic_exchange_memory_order_dispatch(__bound_compare_swap, __memorder, _Sco{});
}
)XXX";
}

#endif // COMPARED_AND_SWAP_H
