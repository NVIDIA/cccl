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

  out << "\n"
      << R"XXX(
#endif // _CCCL_CUDA_COMPILATION()

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_exchange {
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __old;
  __unv<_Type> __new;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Sco) {
    ::cuda::std::__cuda_atomic_exchange(__backend, __ptr, *__old, __new, __order, _Tag{}, _Sco{});
  }
};
template <class _Backend, class _Type, class _Sco>
_CCCL_HOST_DEVICE_API void __cuda_atomic_exchange_dispatch(
  _Backend __backend,
  _Type* __ptr,
  __unv<_Type>& __old,
  __unv<_Type> __new,
  memory_order __order,
  _Sco __scope)
{
  using __value_type _CCCL_NODEBUG    = __unv<_Type>;
  using __proxy_t _CCCL_NODEBUG       = __cuda_atomic_deduce_bitwise_t<__value_type>;
  using __proxy_pointee _CCCL_NODEBUG = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag _CCCL_NODEBUG     = __cuda_atomic_deduce_bitwise_tag_t<__value_type>;
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __old_proxy = reinterpret_cast<__proxy_t*>(&__old);
  __proxy_t* __new_proxy  = reinterpret_cast<__proxy_t*>(&__new);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if(::cuda::std::__cuda_atomic_exchange_weak_if_local(__ptr_proxy, __new_proxy, __old_proxy)) {return;}
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_exchange<_Backend, __proxy_pointee> __bound_swap{
    __backend, __ptr_proxy, __old_proxy, *__new_proxy};
  __cuda_atomic_exchange_order_dispatch(__backend, __bound_swap, __order, __scope, __proxy_tag{});
}

template <class _Backend, class _Type, class _Up, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type> __cuda_atomic_exchange_dispatch(
  _Backend __backend, _Type* __ptr, _Up __new, memory_order __order, _Sco __scope)
{
  using __value_type _CCCL_NODEBUG = __unv<_Type>;
  __value_type __old;
  ::cuda::std::__cuda_atomic_exchange_dispatch(
    __backend, __ptr, __old, static_cast<__value_type>(__new), __order, __scope);
  return __old;
}

#if _CCCL_CUDA_COMPILATION()
)XXX";
}

#endif // EXCHANGE_H
