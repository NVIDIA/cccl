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

#include <string>

#include "definitions.h"
#include <fmt/format.h>

inline void FormatExchange(std::ostream& out)
{
  out << R"XXX(
template <class _Fn, class _Sco>
static inline _CCCL_DEVICE void __cuda_atomic_exchange_memory_order_dispatch(_Fn& __cuda_exch, int __memorder, _Sco) {
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: __cuda_atomic_fence(_Sco{}, __atomic_cuda_seq_cst{}); [[fallthrough]];
        case __ATOMIC_CONSUME: [[fallthrough]];
        case __ATOMIC_ACQUIRE: __cuda_exch(__atomic_cuda_acquire{}); break;
        case __ATOMIC_ACQ_REL: __cuda_exch(__atomic_cuda_acq_rel{}); break;
        case __ATOMIC_RELEASE: __cuda_exch(__atomic_cuda_release{}); break;
        case __ATOMIC_RELAXED: __cuda_exch(__atomic_cuda_relaxed{}); break;
        default: _CCCL_ASSERT(false, "invalid memory order");
      }
    ),
    NV_IS_DEVICE, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: [[fallthrough]];
        case __ATOMIC_ACQ_REL: __cuda_atomic_membar(_Sco{}); [[fallthrough]];
        case __ATOMIC_CONSUME: [[fallthrough]];
        case __ATOMIC_ACQUIRE: __cuda_exch(__atomic_cuda_volatile{}); __cuda_atomic_membar(_Sco{}); break;
        case __ATOMIC_RELEASE: __cuda_atomic_membar(_Sco{}); __cuda_exch(__atomic_cuda_volatile{}); break;
        case __ATOMIC_RELAXED: __cuda_exch(__atomic_cuda_volatile{}); break;
        default: _CCCL_ASSERT(false, "invalid memory order");
      }
    )
  )
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
static inline _CCCL_DEVICE void __cuda_atomic_exchange(
  _Type* __ptr, _Type& __old, _Type __new, {4}, __atomic_cuda_operand_{0}{1}, {6})
{{
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

  const std::string asm_intrinsic_format = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_exchange(
  _Type* __ptr, _Type& __old, _Type __new, {4}, __atomic_cuda_operand_{0}{1}, {6})
{{ asm volatile("atom.exch{3}{5}.{0}{1} %0,[%1],%2;" : "={2}"(__old) : "l"(__ptr), "{2}"(__new) : "memory"); }})XXX";

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

  out << "\n"
      << R"XXX(
template <typename _Type, typename _Tag, typename _Sco>
struct __cuda_atomic_bind_exchange {
  _Type* __ptr;
  _Type* __old;
  _Type* __new;

  template <typename _Atomic_Memorder>
  inline _CCCL_DEVICE void operator()(_Atomic_Memorder) {
    __cuda_atomic_exchange(__ptr, *__old, *__new, _Atomic_Memorder{}, _Tag{}, _Sco{});
  }
};
template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_exchange_cuda(_Type* __ptr, _Type& __old, _Type __new, int __memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __old_proxy = reinterpret_cast<__proxy_t*>(&__old);
  __proxy_t* __new_proxy  = reinterpret_cast<__proxy_t*>(&__new);
  if(__cuda_exchange_weak_if_local(__ptr_proxy, __new_proxy, __old_proxy)) {{return;}}
  __cuda_atomic_bind_exchange<__proxy_t, __proxy_tag, _Sco> __bound_swap{__ptr_proxy, __old_proxy, __new_proxy};
  __cuda_atomic_exchange_memory_order_dispatch(__bound_swap, __memorder, _Sco{});
}
template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_exchange_cuda(_Type volatile* __ptr, _Type& __old, _Type __new, int __memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __old_proxy = reinterpret_cast<__proxy_t*>(&__old);
  __proxy_t* __new_proxy  = reinterpret_cast<__proxy_t*>(&__new);
  if(__cuda_exchange_weak_if_local(__ptr_proxy, __new_proxy, __old_proxy)) {{return;}}
  __cuda_atomic_bind_exchange<__proxy_t, __proxy_tag, _Sco> __bound_swap{__ptr_proxy, __old_proxy, __new_proxy};
  __cuda_atomic_exchange_memory_order_dispatch(__bound_swap, __memorder, _Sco{});
}
)XXX";
}

#endif // EXCHANGE_H
