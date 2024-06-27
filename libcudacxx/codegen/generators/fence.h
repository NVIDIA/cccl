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

#include <string>

#include "definitions.h"
#include <fmt/format.h>

static std::string membar_scope(Scope sco)
{
  static std::map scope_map{
    std::pair{Scope::GPU, ".gl"},
    std::pair{Scope::System, ".sys"},
    std::pair{Scope::CTA, ".cta"},
  };

  return scope_map[sco];
}

static void FormatFence(std::ostream& out)
{
  // Argument ID Reference
  // 0 - Membar scope tag
  // 1 - Membar scope
  const std::string intrinsic_membar = R"XXX(
static inline _CCCL_DEVICE void __cuda_atomic_membar({0})
{{ asm volatile("membar{1};" ::: "memory"); }})XXX";

  const std::map membar_scopes{
    std::pair{Scope::GPU, ".gl"},
    std::pair{Scope::System, ".sys"},
    std::pair{Scope::CTA, ".cta"},
  };

  for (const auto& sco : membar_scopes)
  {
    out << fmt::format(intrinsic_membar, scope_tag(sco.first), sco.second);
  }

  // Argument ID Reference
  // 0 - Fence scope tag
  // 1 - Fence scope
  // 2 - Fence order tag
  // 3 - Fence order
  const std::string intrinsic_fence = R"XXX(
static inline _CCCL_DEVICE void __cuda_atomic_fence({0}, __atomic_cuda_{2})
{{ asm volatile("fence{1}{3};" ::: "memory"); }})XXX";

  const std::array fence_scopes{
    Scope::CTA,
    Scope::Cluster,
    Scope::GPU,
    Scope::System,
  };

  const std::array fence_semantics{
    Semantic::Acq_Rel,
    Semantic::Seq_Cst,
  };

  for (const auto& sco : fence_scopes)
  {
    for (const auto& sem : fence_semantics)
    {
      out << fmt::format(intrinsic_fence, scope_tag(sco), scope(sco), semantic_tag(sem), semantic(sem));
    }
  }
  out << "\n";
}

#endif // FENCE_H
