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
  // 0 - Atomic Operation
  // 1 - Operand type constraint
  // 2 - Operand transform
  constexpr auto fetch_bind_invoke = R"XXX(
#endif // _CCCL_CUDA_COMPILATION()

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_fetch_{0} {{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __dst;
  __unv<_Type> __op;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Sco) {{
    ::cuda::std::__cuda_atomic_fetch_{0}(__backend, __ptr, *__dst, __op, __order, _Tag{{}}, _Sco{{}});
  }}
}};
template <class _Backend,
          class _Type,
          class _Up,
          class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type> __cuda_atomic_fetch_{0}_dispatch(
  _Backend __backend, _Type* __ptr, _Up __op, memory_order __order, _Sco __scope)
{{{2}
  using __value_type     = __unv<_Type>;
  using __proxy_t        = __cuda_atomic_deduce_{1}_t<__value_type>;
  using __proxy_pointee  = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag      = __cuda_atomic_deduce_{1}_tag_t<__value_type>;
  __value_type __dst{{}};
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {{
    if (::cuda::std::__cuda_atomic_fetch_{0}_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy)) {{return __dst;}}
  }}
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_fetch_{0}<_Backend, __proxy_pointee> __bound_{0}{{
    __backend, __ptr_proxy, __dst_proxy, *__op_proxy}};
  __cuda_atomic_fetch_order_dispatch(__backend, __bound_{0}, __order, __scope, __proxy_tag{{}});
  return __dst;
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

  bool first_op = true;
  for (auto& op_kp : op_support_map)
  {
    if (!first_op)
    {
      out << "\n#if _CCCL_CUDA_COMPILATION()\n";
    }
    first_op = false;

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
    out << "\n" << std::format(fetch_bind_invoke, op_name, deduction, fetch_op_transform(op_name));
  }
}

#endif // FETCH_OPS_H
