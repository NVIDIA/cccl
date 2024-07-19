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

inline std::string semantic_ld_st(Semantic sem)
{
  static std::map sem_map = {
    std::pair{Semantic::Relaxed, ".relaxed"},
    std::pair{Semantic::Release, ".release"},
    std::pair{Semantic::Acquire, ".acquire"},
    std::pair{Semantic::Volatile, ".volatile"},
  };
  return sem_map[sem];
}

inline std::string scope_ld_st(Semantic sem, Scope sco)
{
  if (sem == Semantic::Volatile)
  {
    return "";
  }
  return scope(sco);
}

inline void FormatLoad(std::ostream& out)
{
  out << R"XXX(
template <class _Fn, class _Sco>
static inline _CCCL_DEVICE void __cuda_atomic_load_memory_order_dispatch(_Fn &__cuda_load, int __memorder, _Sco) {
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: __cuda_atomic_fence(_Sco{}, __atomic_cuda_seq_cst{}); _CCCL_FALLTHROUGH();
        case __ATOMIC_CONSUME: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQUIRE: __cuda_load(__atomic_cuda_acquire{}); break;
        case __ATOMIC_RELAXED: __cuda_load(__atomic_cuda_relaxed{}); break;
        default: assert(0);
      }
    ),
    NV_IS_DEVICE, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: __cuda_atomic_membar(_Sco{}); _CCCL_FALLTHROUGH();
        case __ATOMIC_CONSUME: _CCCL_FALLTHROUGH();
        case __ATOMIC_ACQUIRE: __cuda_load(__atomic_cuda_volatile{}); __cuda_atomic_membar(_Sco{}); break;
        case __ATOMIC_RELAXED: __cuda_load(__atomic_cuda_volatile{}); break;
        default: assert(0);
      }
    )
  )
}
)XXX";

  // Argument ID Reference
  // 0 - Operand Type
  // 1 - Operand Size
  // 2 - Constraint
  // 3 - Memory order
  // 4 - Memory order semantic
  // 5 - Scope tag
  // 6 - Scope semantic
  // 7 - Mmio tag
  // 8 - Mmio semantic
  const std::string asm_intrinsic_format_128 = R"XXX(
  template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_load(
  const _Type* __ptr, _Type& __dst, {3}, __atomic_cuda_operand_{0}{1}, {5}, {7})
{{
  asm volatile(R"YYY(
    .reg .b128 _d;
    ld{8}{4}{6}.b128 [%2],_d;
    mov.b128 _d, {{%0, %1}};
)YYY" : "=l"(__dst.__x),"=l"(__dst.__y) : "l"(__ptr) : "memory");
}})XXX";
  const std::string asm_intrinsic_format     = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_load(
  const _Type* __ptr, _Type& __dst, {3}, __atomic_cuda_operand_{0}{1}, {5}, {7})
{{ asm volatile("ld{8}{4}{6}.{0}{1} %0,[%1];" : "={2}"(__dst) : "l"(__ptr) : "memory"); }})XXX";

  constexpr size_t supported_sizes[] = {
    16,
    32,
    64,
    128,
  };

  constexpr Operand supported_types[] = {
    Operand::Bit,
    Operand::Floating,
    Operand::Unsigned,
    Operand::Signed,
  };

  constexpr Semantic supported_semantics[] = {
    Semantic::Acquire,
    Semantic::Relaxed,
    Semantic::Volatile,
  };

  constexpr Scope supported_scopes[] = {
    Scope::CTA,
    Scope::Cluster,
    Scope::GPU,
    Scope::System,
  };

  constexpr Mmio mmio_states[] = {
    Mmio::Disabled,
    Mmio::Enabled,
  };

  for (auto size : supported_sizes)
  {
    for (auto type : supported_types)
    {
      for (auto sem : supported_semantics)
      {
        for (auto sco : supported_scopes)
        {
          for (auto mm : mmio_states)
          {
            if (size == 16 && type == Operand::Floating)
            {
              continue;
            }
            if (size == 128 && type != Operand::Bit)
            {
              continue;
            }
            if ((mm == Mmio::Enabled) && ((sco != Scope::System) || (sem != Semantic::Relaxed)))
            {
              continue;
            }
            out << fmt::format(
              (size == 128) ? asm_intrinsic_format_128 : asm_intrinsic_format,
              /* 0 */ operand(type),
              /* 1 */ size,
              /* 2 */ constraints(type, size),
              /* 3 */ semantic_tag(sem),
              /* 4 */ semantic_ld_st(sem),
              /* 5 */ scope_tag(sco),
              /* 6 */ scope_ld_st(sem, sco),
              /* 7 */ mmio_tag(mm),
              /* 8 */ mmio(mm));
          }
        }
      }
    }
  }
  out << "\n"
      << R"XXX(
template <typename _Type, typename _Tag, typename _Sco, typename _Mmio>
struct __cuda_atomic_bind_load {
  const _Type* __ptr;
  _Type* __dst;

  template <typename _Atomic_Memorder>
  inline _CCCL_DEVICE void operator()(_Atomic_Memorder) {
    __cuda_atomic_load(__ptr, *__dst, _Atomic_Memorder{}, _Tag{}, _Sco{}, _Mmio{});
  }
};
template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_load_cuda(const _Type* __ptr, _Type& __dst, int __memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  const __proxy_t* __ptr_proxy = reinterpret_cast<const __proxy_t*>(__ptr);
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __cuda_atomic_bind_load<__proxy_t, __proxy_tag, _Sco, __atomic_cuda_mmio_disable> __bound_load{__ptr_proxy, __dst_proxy};
  __cuda_atomic_load_memory_order_dispatch(__bound_load, __memorder, _Sco{});
}
template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_load_cuda(const _Type volatile* __ptr, _Type& __dst, int __memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  const __proxy_t* __ptr_proxy = reinterpret_cast<const __proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __cuda_atomic_bind_load<__proxy_t, __proxy_tag, _Sco, __atomic_cuda_mmio_disable> __bound_load{__ptr_proxy, __dst_proxy};
  __cuda_atomic_load_memory_order_dispatch(__bound_load, __memorder, _Sco{});
}
)XXX";
}

inline void FormatStore(std::ostream& out)
{
  out << R"XXX(
template <class _Fn, class _Sco>
static inline _CCCL_DEVICE void __cuda_atomic_store_memory_order_dispatch(_Fn &__cuda_store, int __memorder, _Sco) {
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_RELEASE: __cuda_store(__atomic_cuda_release{}); break;
        case __ATOMIC_SEQ_CST: __cuda_atomic_fence(_Sco{}, __atomic_cuda_seq_cst{}); _CCCL_FALLTHROUGH();
        case __ATOMIC_RELAXED: __cuda_store(__atomic_cuda_relaxed{}); break;
        default: assert(0);
      }
    ),
    NV_IS_DEVICE, (
      switch (__memorder) {
        case __ATOMIC_RELEASE: _CCCL_FALLTHROUGH();
        case __ATOMIC_SEQ_CST: __cuda_atomic_membar(_Sco{}); _CCCL_FALLTHROUGH();
        case __ATOMIC_RELAXED: __cuda_store(__atomic_cuda_volatile{}); break;
        default: assert(0);
      }
    )
  )
}
)XXX";
  // Argument ID Reference
  // 0 - Operand Type
  // 1 - Operand Size
  // 2 - Constraint
  // 3 - Memory order
  // 4 - Memory order semantic
  // 5 - Scope tag
  // 6 - Scope semantic
  // 7 - Mmio tag
  // 8 - Mmio semantic
  const std::string asm_intrinsic_format_128 = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_store(
  _Type* __ptr, _Type& __val, {3}, __atomic_cuda_operand_{0}{1}, {5}, {7})
{{
  asm volatile(R"YYY(
    .reg .b128 _v;
    mov.b128 {{%1, %2}}, _v;
    st{8}{4}{6}.b128 [%0],_v;
)YYY" :: "l"(__ptr), "l"(__val.__x),"l"(__val.__y) : "memory");
}})XXX";
  const std::string asm_intrinsic_format     = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_store(
  _Type* __ptr, _Type& __val, {3}, __atomic_cuda_operand_{0}{1}, {5}, {7})
{{ asm volatile("st{8}{4}{6}.{0}{1} [%0],%1;" :: "l"(__ptr), "{2}"(__val) : "memory"); }})XXX";

  constexpr size_t supported_sizes[] = {
    16,
    32,
    64,
    128,
  };

  constexpr Operand supported_types[] = {
    Operand::Bit,
  };

  constexpr Semantic supported_semantics[] = {
    Semantic::Release,
    Semantic::Relaxed,
    Semantic::Volatile,
  };

  constexpr Scope supported_scopes[] = {
    Scope::CTA,
    Scope::Cluster,
    Scope::GPU,
    Scope::System,
  };

  constexpr Mmio mmio_states[] = {
    Mmio::Disabled,
    Mmio::Enabled,
  };

  for (auto size : supported_sizes)
  {
    for (auto type : supported_types)
    {
      for (auto sem : supported_semantics)
      {
        for (auto sco : supported_scopes)
        {
          for (auto mm : mmio_states)
          {
            if (size == 16 && type == Operand::Floating)
            {
              continue;
            }
            if (size == 128 && type != Operand::Bit)
            {
              continue;
            }
            if ((mm == Mmio::Enabled) && ((sco != Scope::System) || (sem != Semantic::Relaxed)))
            {
              continue;
            }
            out << fmt::format(
              (size == 128) ? asm_intrinsic_format_128 : asm_intrinsic_format,
              /* 0 */ operand(type),
              /* 1 */ size,
              /* 2 */ constraints(type, size),
              /* 3 */ semantic_tag(sem),
              /* 4 */ semantic_ld_st(sem),
              /* 5 */ scope_tag(sco),
              /* 6 */ scope_ld_st(sem, sco),
              /* 7 */ mmio_tag(mm),
              /* 8 */ mmio(mm));
          }
        }
      }
    }
  }
  out << "\n"
      << R"XXX(
template <typename _Type, typename _Tag, typename _Sco, typename _Mmio>
struct __cuda_atomic_bind_store {
  _Type* __ptr;
  _Type* __val;

  template <typename _Atomic_Memorder>
  inline _CCCL_DEVICE void operator()(_Atomic_Memorder) {
    __cuda_atomic_store(__ptr, *__val, _Atomic_Memorder{}, _Tag{}, _Sco{}, _Mmio{});
  }
};
template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_store_cuda(_Type* __ptr, _Type& __val, int __memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __val_proxy = reinterpret_cast<__proxy_t*>(&__val);
  __cuda_atomic_bind_store<__proxy_t, __proxy_tag, _Sco, __atomic_cuda_mmio_disable> __bound_store{__ptr_proxy, __val_proxy};
  __cuda_atomic_store_memory_order_dispatch(__bound_store, __memorder, _Sco{});
}
template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_store_cuda(volatile _Type* __ptr, _Type& __val, int __memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __val_proxy = reinterpret_cast<__proxy_t*>(&__val);
  __cuda_atomic_bind_store<__proxy_t, __proxy_tag, _Sco, __atomic_cuda_mmio_disable> __bound_store{__ptr_proxy, __val_proxy};
  __cuda_atomic_store_memory_order_dispatch(__bound_store, __memorder, _Sco{});
}
)XXX";
}

#endif // LD_ST_H
