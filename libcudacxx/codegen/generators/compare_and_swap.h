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

#include <format>
#include <string>

#include "definitions.h"

inline void FormatCompareAndSwap(std::ostream& out)
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
static inline _CCCL_DEVICE bool __cuda_atomic_compare_exchange(
  __cuda_atomic_ptx_backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __cmp, __unv<_Type> __op, __cuda_atomic_cas_strong, {4} __order, __cuda_atomic_operand_{0}{1}, {6})
{{
  ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {6}{{}});
  static_assert(__cccl_ptx_isa >= 840 && (sizeof(_Type) == 16), "128b CAS is not supported until PTX ISA version 840");
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_90, (),
    NV_ANY_TARGET, (__atomic_cas_128b_unsupported_before_SM_90();)
  )
  asm volatile(R"YYY(
    {{
      .reg .b128 _d;
      .reg .b128 _v;
      mov.b128 _d, {{%3, %4}};
      mov.b128 _v, {{%5, %6}};
      atom.cas{3}{5}.b128 _d,[%2],_d,_v;
      mov.b128 {{%0, %1}}, _d;
    }}
  )YYY" : "=l"(__dst.__x),"=l"(__dst.__y) : "l"(__ptr), "l"(__cmp.__x),"l"(__cmp.__y), "l"(__op.__x),"l"(__op.__y) : "memory"); return __dst.__x == __cmp.__x && __dst.__y == __cmp.__y; }})XXX";
  constexpr auto asm_intrinsic_format     = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE bool __cuda_atomic_compare_exchange(
  __cuda_atomic_ptx_backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __cmp, __unv<_Type> __op, __cuda_atomic_cas_strong, {4} __order, __cuda_atomic_operand_{0}{1}, {6})
{{ ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {6}{{}}); asm volatile("atom.cas{3}{5}.{0}{1} %0,[%1],%2,%3;" : "={2}"(__dst) : "l"(__ptr), "{2}"(__cmp), "{2}"(__op) : "memory"); return __dst == __cmp; }})XXX";

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

  out << "\n"
      << R"XXX(
#endif // _CCCL_CUDA_COMPILATION()

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_compare_exchange {
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __exp;
  __unv<_Type> __cmp;
  __unv<_Type> __des;

  template <typename _Atomic_Memorder, typename _Cas, typename _Tag, typename _Sco>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool operator()(_Atomic_Memorder __order, _Cas, _Tag, _Sco) {
    return ::cuda::std::__cuda_atomic_compare_exchange(
      __backend, __ptr, *__exp, __cmp, __des, _Cas{}, __order, _Tag{}, _Sco{});
  }
};
template <class _Backend, class _Type, class _Cas, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API bool __cuda_atomic_compare_exchange_dispatch(
  _Backend __backend,
  _Type* __ptr,
  __unv<_Type>* __exp,
  __unv<_Type> __des,
  _Cas,
  memory_order __success,
  memory_order __failure,
  _Sco __scope)
{
  using __value_type     = __unv<_Type>;
  using __proxy_t        = __cuda_atomic_deduce_bitwise_t<__value_type>;
  using __proxy_pointee  = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag      = __cuda_atomic_deduce_bitwise_tag_t<__value_type>;
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __exp_proxy = reinterpret_cast<__proxy_t*>(__exp);
  __proxy_t* __des_proxy  = reinterpret_cast<__proxy_t*>(&__des);
  bool __res = false;
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_compare_exchange_weak_if_local(__ptr_proxy, __exp_proxy, __des_proxy, &__res)) {return __res;}
  }
  __cuda_atomic_bind_compare_exchange<_Backend, __proxy_pointee> __bound_compare_swap{
    __backend, __ptr_proxy, __exp_proxy, *__exp_proxy, *__des_proxy};
  return __cuda_atomic_compare_exchange_order_dispatch(
    __backend, __bound_compare_swap, __success, __failure, __scope, _Cas{}, __proxy_tag{});
}

#if _CCCL_CUDA_COMPILATION()
)XXX";
}

#endif // COMPARED_AND_SWAP_H
