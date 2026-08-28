//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef FENCE_H
#define FENCE_H

#include <format>
#include <string>

#include "definitions.h"

inline std::string membar_scope(Scope sco)
{
  static std::map scope_map{
    std::pair{Scope::GPU, ".gl"},
    std::pair{Scope::System, ".sys"},
    std::pair{Scope::CTA, ".cta"},
  };

  return scope_map[sco];
}

inline void FormatFence(std::ostream& out)
{
  // Argument ID Reference
  // 0 - Membar scope tag
  // 1 - Membar scope
  constexpr auto intrinsic_membar = R"XXX(
_CCCL_DEVICE_API inline void __cuda_atomic_membar({0})
{{ asm volatile("membar{1};" ::: "memory"); }})XXX";

  const std::map membar_scopes{
    std::pair{Scope::GPU, ".gl"},
    std::pair{Scope::System, ".sys"},
    std::pair{Scope::CTA, ".cta"},
  };

  for (const auto& sco : membar_scopes)
  {
    out << std::format(intrinsic_membar, scope_tag(sco.first), sco.second);
  }

  // Argument ID Reference
  // 0 - Fence scope tag
  // 1 - Fence scope
  // 2 - Fence order tag
  // 3 - Fence order
  constexpr auto intrinsic_fence = R"XXX(
_CCCL_DEVICE_API inline void __cuda_atomic_fence({0}, {2})
{{ asm volatile("fence{1}{3};" ::: "memory"); }})XXX";

  const Scope fence_scopes[] = {
    Scope::CTA,
    Scope::Cluster,
    Scope::GPU,
    Scope::System,
  };

  const Semantic fence_semantics[] = {
    Semantic::Acq_Rel,
    Semantic::Seq_Cst,
  };

  for (const auto& sco : fence_scopes)
  {
    for (const auto& sem : fence_semantics)
    {
      out << std::format(intrinsic_fence, scope_tag(sco), semantic(sem), semantic_tag(sem), scope(sco));
    }
  }
  out << "\n"
      << R"XXX(
template <class _Order, class _Sco>
_CCCL_DEVICE_API void
__cuda_atomic_ptx_maybe_sc_fence(__cuda_atomic_ptx_order<_Order> __order, _Sco __scope)
{
  if (__order.__was_seq_cst)
  {
    ::cuda::std::__cuda_atomic_fence(__scope, __cuda_atomic_order_seq_cst{});
  }
}

template <class _Sco>
_CCCL_DEVICE_API void __cuda_atomic_ptx_maybe_sc_fence(__cuda_atomic_order_volatile, _Sco)
{}

template <typename _Sco>
_CCCL_DEVICE_API void __cuda_atomic_thread_fence(
  __cuda_atomic_ptx_backend, memory_order __order, _Sco) {
  [[maybe_unused]] const int __memorder = __atomic_order_to_int(__order);
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: ::cuda::std::__cuda_atomic_fence(_Sco{}, __cuda_atomic_order_seq_cst{}); break;
        case __ATOMIC_CONSUME: [[fallthrough]];
        case __ATOMIC_ACQUIRE: [[fallthrough]];
        case __ATOMIC_ACQ_REL: [[fallthrough]];
        case __ATOMIC_RELEASE: ::cuda::std::__cuda_atomic_fence(_Sco{}, __cuda_atomic_order_acq_rel{}); break;
        case __ATOMIC_RELAXED: break;
        default: _CCCL_ASSERT(false, "invalid memory order");
      }
    ),
    NV_IS_DEVICE, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: [[fallthrough]];
        case __ATOMIC_CONSUME: [[fallthrough]];
        case __ATOMIC_ACQUIRE: [[fallthrough]];
        case __ATOMIC_ACQ_REL: [[fallthrough]];
        case __ATOMIC_RELEASE: ::cuda::std::__cuda_atomic_membar(_Sco{}); break;
        case __ATOMIC_RELAXED: break;
        default: _CCCL_ASSERT(false, "invalid memory order");
      }
    )
  )
}
)XXX";
}

#endif // FENCE_H
