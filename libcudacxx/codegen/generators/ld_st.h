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
_CCCL_DEVICE_API void __cuda_atomic_load(
  __cuda_atomic_ptx_backend, const _Type* __ptr, __unv<_Type>& __dst, {3} __order, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{
  ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {5}{{}});
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
_CCCL_DEVICE_API void __cuda_atomic_load(
  __cuda_atomic_ptx_backend, const _Type* __ptr, __unv<_Type>& __dst, {3} __order, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{ ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {5}{{}}); asm volatile("ld{8}{4}{6}.{0}{1} %0,[%1];" : "={2}"(__dst) : "l"(__ptr) : "memory"); }})XXX";
  constexpr auto asm_intrinsic_format_8   = R"XXX(
template <class _Type>
_CCCL_DEVICE_API void __cuda_atomic_load(
  __cuda_atomic_ptx_backend, const _Type* __ptr, __unv<_Type>& __dst, {3} __order, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{
  ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {5}{{}});
  uint16_t __tmp;
  asm volatile("ld{8}{4}{6}.{0}{1} %0,[%1];" : "={2}"(__tmp) : "l"(__ptr) : "memory");
  __dst = static_cast<__unv<_Type>>(__tmp);
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
                /* 3 */ ptx_semantic_tag(sem),
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
                /* 3 */ ptx_semantic_tag(sem),
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
                /* 3 */ ptx_semantic_tag(sem),
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
  out << "\n";
}

inline void FormatStore(std::ostream& out)
{
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
_CCCL_DEVICE_API void __cuda_atomic_store(
  __cuda_atomic_ptx_backend, _Type* __ptr, __unv<_Type> __val, {3} __order, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{
  ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {5}{{}});
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
_CCCL_DEVICE_API void __cuda_atomic_store(
  __cuda_atomic_ptx_backend, _Type* __ptr, __unv<_Type> __val, {3} __order, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{ ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {5}{{}}); asm volatile("st{8}{4}{6}.{0}{1} [%0],%1;" :: "l"(__ptr), "{2}"(__val) : "memory"); }})XXX";
  constexpr auto asm_intrinsic_format_8   = R"XXX(
template <class _Type>
_CCCL_DEVICE_API void __cuda_atomic_store(
  __cuda_atomic_ptx_backend, _Type* __ptr, __unv<_Type> __val, {3} __order, __cuda_atomic_operand_{0}{1}, {5}, {7})
{{
  ::cuda::std::__cuda_atomic_ptx_maybe_sc_fence(__order, {5}{{}});
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
                /* 3 */ ptx_semantic_tag(sem),
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
                /* 3 */ ptx_semantic_tag(sem),
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
                /* 3 */ ptx_semantic_tag(sem),
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
  out << "\n";
}

#endif // LD_ST_H
