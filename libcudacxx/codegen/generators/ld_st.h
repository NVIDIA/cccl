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

#include <format>
#include <string>

#include "definitions.h"

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
static inline _CCCL_DEVICE void __cuda_atomic_load_order_dispatch(
  __cuda_atomic_ptx_backend, _Fn& __cuda_load, memory_order __order, _Sco) {
  const int __memorder = __atomic_order_to_int(__order);
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: __cuda_atomic_fence(_Sco{}, __cuda_atomic_order_seq_cst{}); [[fallthrough]];
        case __ATOMIC_CONSUME: [[fallthrough]];
        case __ATOMIC_ACQUIRE: __cuda_load(__cuda_atomic_order_acquire{}); break;
        case __ATOMIC_RELAXED: __cuda_load(__cuda_atomic_order_relaxed{}); break;
        default: _CCCL_ASSERT(false, "invalid memory order");
      }
    ),
    NV_IS_DEVICE, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: __cuda_atomic_membar(_Sco{}); [[fallthrough]];
        case __ATOMIC_CONSUME: [[fallthrough]];
        case __ATOMIC_ACQUIRE: __cuda_load(__cuda_atomic_order_volatile{}); __cuda_atomic_membar(_Sco{}); break;
        case __ATOMIC_RELAXED: __cuda_load(__cuda_atomic_order_volatile{}); break;
        default: _CCCL_ASSERT(false, "invalid memory order");
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
  constexpr auto asm_intrinsic_format_128 = R"XXX(
  template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_load(
  __cuda_atomic_ptx_backend, const _Type* __ptr, _Type& __dst, {3}, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{
  static_assert(__cccl_ptx_isa >= 840 && (sizeof(_Type) == 16), "128b ld/st is not supported until PTX ISA version 840");
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (),
    NV_ANY_TARGET, (__atomic_ldst_128b_unsupported_before_SM_70();)
  )
  asm volatile(R"YYY(
    {{
      .reg .b128 _d;
      ld{8}{4}{6}.b128 _d,[%2];
      mov.b128 {{%0, %1}}, _d;
    }}
  )YYY" : "=l"(__dst.__x),"=l"(__dst.__y) : "l"(__ptr) : "memory");
}})XXX";
  constexpr auto asm_intrinsic_format     = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_load(
  __cuda_atomic_ptx_backend, const _Type* __ptr, _Type& __dst, {3}, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{ asm volatile("ld{8}{4}{6}.{0}{1} %0,[%1];" : "={2}"(__dst) : "l"(__ptr) : "memory"); }})XXX";
  constexpr auto asm_intrinsic_format_8   = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_load(
  __cuda_atomic_ptx_backend, const _Type* __ptr, _Type& __dst, {3}, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{
  uint16_t __tmp;
  asm volatile("ld{8}{4}{6}.{0}{1} %0,[%1];" : "={2}"(__tmp) : "l"(__ptr) : "memory");
  __dst = static_cast<_Type>(__tmp);
}})XXX";

  constexpr size_t supported_sizes[] = {
    8,
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
            if (size <= 16 && type == Operand::Floating)
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

            if (size == 128)
            {
              out << std::format(
                asm_intrinsic_format_128,
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
            else if (size == 8)
            {
              out << std::format(
                asm_intrinsic_format_8,
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
            else
            {
              out << std::format(
                asm_intrinsic_format,
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
  }
  out << "\n"
      << R"XXX(
#endif // _CCCL_CUDA_COMPILATION()

template <typename _Backend, typename _Type, typename _Tag, typename _Sco, typename _Mmio>
struct __cuda_atomic_bind_load {
  _Backend __backend;
  const _Type* __ptr;
  _Type* __dst;

  template <typename _Atomic_Memorder>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order) {
    __cuda_atomic_load(__backend, __ptr, *__dst, __order, _Tag{}, _Sco{}, _Mmio{});
  }
};
template <class _Backend, class _Type, class _Sco>
_CCCL_HOST_DEVICE_API void
__cuda_atomic_load_dispatch(_Backend __backend, const _Type* __ptr, _Type& __dst, memory_order __order, _Sco __scope)
{
  using __proxy_t        = typename __cuda_atomic_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __cuda_atomic_deduce_bitwise<_Type>::__tag;
  const __proxy_t* __ptr_proxy = reinterpret_cast<const __proxy_t*>(__ptr);
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (__cuda_atomic_load_weak_if_local(__ptr_proxy, __dst_proxy, sizeof(__proxy_t))) {return;}
  }
  __cuda_atomic_bind_load<_Backend, __proxy_t, __proxy_tag, _Sco, __cuda_atomic_mmio_disable> __bound_load{
    __backend, __ptr_proxy, __dst_proxy};
  __cuda_atomic_load_order_dispatch(__backend, __bound_load, __order, __scope);
}
template <class _Backend, class _Type, class _Sco>
_CCCL_HOST_DEVICE_API void __cuda_atomic_load_dispatch(
  _Backend __backend, const _Type volatile* __ptr, _Type& __dst, memory_order __order, _Sco __scope)
{
  using __proxy_t        = typename __cuda_atomic_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __cuda_atomic_deduce_bitwise<_Type>::__tag;
  const __proxy_t* __ptr_proxy = reinterpret_cast<const __proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (__cuda_atomic_load_weak_if_local(__ptr_proxy, __dst_proxy, sizeof(__proxy_t))) {return;}
  }
  __cuda_atomic_bind_load<_Backend, __proxy_t, __proxy_tag, _Sco, __cuda_atomic_mmio_disable> __bound_load{
    __backend, __ptr_proxy, __dst_proxy};
  __cuda_atomic_load_order_dispatch(__backend, __bound_load, __order, __scope);
}

template <class _Backend, class _Type, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type
__cuda_atomic_load_dispatch(_Backend __backend, const _Type* __ptr, memory_order __order, _Sco __scope)
{
  _Type __dst;
  __cuda_atomic_load_dispatch(__backend, __ptr, __dst, __order, __scope);
  return __dst;
}

template <class _Backend, class _Type, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type
__cuda_atomic_load_dispatch(_Backend __backend, const volatile _Type* __ptr, memory_order __order, _Sco __scope)
{
  _Type __dst;
  __cuda_atomic_load_dispatch(__backend, __ptr, __dst, __order, __scope);
  return __dst;
}

#if _CCCL_CUDA_COMPILATION()
)XXX";
}

inline void FormatStore(std::ostream& out)
{
  out << R"XXX(
template <class _Fn, class _Sco>
static inline _CCCL_DEVICE void __cuda_atomic_store_order_dispatch(
  __cuda_atomic_ptx_backend, _Fn& __cuda_store, memory_order __order, _Sco) {
  const int __memorder = __atomic_order_to_int(__order);
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_RELEASE: __cuda_store(__cuda_atomic_order_release{}); break;
        case __ATOMIC_SEQ_CST: __cuda_atomic_fence(_Sco{}, __cuda_atomic_order_seq_cst{}); [[fallthrough]];
        case __ATOMIC_RELAXED: __cuda_store(__cuda_atomic_order_relaxed{}); break;
        default: _CCCL_ASSERT(false, "invalid memory order");
      }
    ),
    NV_IS_DEVICE, (
      switch (__memorder) {
        case __ATOMIC_RELEASE: [[fallthrough]];
        case __ATOMIC_SEQ_CST: __cuda_atomic_membar(_Sco{}); [[fallthrough]];
        case __ATOMIC_RELAXED: __cuda_store(__cuda_atomic_order_volatile{}); break;
        default: _CCCL_ASSERT(false, "invalid memory order");
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
  constexpr auto asm_intrinsic_format_128 = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_store(
  __cuda_atomic_ptx_backend, _Type* __ptr, _Type& __val, {3}, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{
  static_assert(__cccl_ptx_isa >= 840 && (sizeof(_Type) == 16), "128b ld/st is not supported until PTX ISA version 840");
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (),
    NV_ANY_TARGET, (__atomic_ldst_128b_unsupported_before_SM_70();)
  )
  asm volatile(R"YYY(
    {{
      .reg .b128 _v;
      mov.b128 _v, {{%1, %2}};
      st{8}{4}{6}.b128 [%0],_v;
    }}
  )YYY" :: "l"(__ptr), "l"(__val.__x),"l"(__val.__y) : "memory");
}})XXX";
  constexpr auto asm_intrinsic_format     = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_store(
  __cuda_atomic_ptx_backend, _Type* __ptr, _Type& __val, {3}, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{ asm volatile("st{8}{4}{6}.{0}{1} [%0],%1;" :: "l"(__ptr), "{2}"(__val) : "memory"); }})XXX";
  constexpr auto asm_intrinsic_format_8   = R"XXX(
template <class _Type>
static inline _CCCL_DEVICE void __cuda_atomic_store(
  __cuda_atomic_ptx_backend, _Type* __ptr, _Type& __val, {3}, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{
  const uint16_t __tmp = static_cast<uint16_t>(__val);
  asm volatile("st{8}{4}{6}.{0}{1} [%0],%1;" :: "l"(__ptr), "{2}"(__tmp) : "memory");
}})XXX";

  constexpr size_t supported_sizes[] = {
    8,
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

            if (size == 128)
            {
              out << std::format(
                asm_intrinsic_format_128,
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
            else if (size == 8)
            {
              out << std::format(
                asm_intrinsic_format_8,
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
            else
            {
              out << std::format(
                asm_intrinsic_format,
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
  }
  out << "\n"
      << R"XXX(
#endif // _CCCL_CUDA_COMPILATION()

template <typename _Backend, typename _Type, typename _Tag, typename _Sco, typename _Mmio>
struct __cuda_atomic_bind_store {
  _Backend __backend;
  _Type* __ptr;
  _Type* __val;

  template <typename _Atomic_Memorder>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order) {
    __cuda_atomic_store(__backend, __ptr, *__val, __order, _Tag{}, _Sco{}, _Mmio{});
  }
};
template <class _Backend, class _Type, class _Up, class _Sco>
_CCCL_HOST_DEVICE_API void
__cuda_atomic_store_dispatch(_Backend __backend, _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  using __proxy_t        = typename __cuda_atomic_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __cuda_atomic_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  _Type __store           = __val;
  __proxy_t* __val_proxy = reinterpret_cast<__proxy_t*>(&__store);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (__cuda_atomic_store_weak_if_local(__ptr_proxy, __val_proxy, sizeof(__proxy_t))) {return;}
  }
  __cuda_atomic_bind_store<_Backend, __proxy_t, __proxy_tag, _Sco, __cuda_atomic_mmio_disable> __bound_store{
    __backend, __ptr_proxy, __val_proxy};
  __cuda_atomic_store_order_dispatch(__backend, __bound_store, __order, __scope);
}
template <class _Backend, class _Type, class _Up, class _Sco>
_CCCL_HOST_DEVICE_API void __cuda_atomic_store_dispatch(
  _Backend __backend, volatile _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  using __proxy_t        = typename __cuda_atomic_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __cuda_atomic_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  _Type __store           = __val;
  __proxy_t* __val_proxy = reinterpret_cast<__proxy_t*>(&__store);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (__cuda_atomic_store_weak_if_local(__ptr_proxy, __val_proxy, sizeof(__proxy_t))) {return;}
  }
  __cuda_atomic_bind_store<_Backend, __proxy_t, __proxy_tag, _Sco, __cuda_atomic_mmio_disable> __bound_store{
    __backend, __ptr_proxy, __val_proxy};
  __cuda_atomic_store_order_dispatch(__backend, __bound_store, __order, __scope);
}

#if _CCCL_CUDA_COMPILATION()
)XXX";
}

#endif // LD_ST_H
