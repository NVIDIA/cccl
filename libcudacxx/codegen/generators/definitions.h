//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

enum class Mmio
{
  Disabled,
  Enabled,
};

inline std::string mmio(Mmio m)
{
  static const char* mmio_map[]{
    "",
    ".mmio",
  };
  return mmio_map[std::underlying_type_t<Mmio>(m)];
}

inline std::string mmio_tag(Mmio m)
{
  static const char* mmio_map[]{
    "__atomic_cuda_mmio_disable",
    "__atomic_cuda_mmio_enable",
  };
  return mmio_map[std::underlying_type_t<Mmio>(m)];
}

enum class Operand
{
  Floating,
  Unsigned,
  Signed,
  Bit,
};

inline std::string operand(Operand op)
{
  static std::map op_map = {
    std::pair{Operand::Floating, "f"},
    std::pair{Operand::Unsigned, "u"},
    std::pair{Operand::Signed, "s"},
    std::pair{Operand::Bit, "b"},
  };
  return op_map[op];
}

inline std::string operand_proxy_type(Operand op, size_t sz)
{
  if (op == Operand::Floating)
  {
    if (sz == 32)
    {
      return {"float"};
    }
    else
    {
      return {"double"};
    }
  }
  else if (op == Operand::Signed)
  {
    return fmt::format("int{}_t", sz);
  }
  // Binary and unsigned can be the same proxy_type
  return fmt::format("uint{}_t", sz);
}

inline std::string constraints(Operand op, size_t sz)
{
  static std::map constraint_map = {
    std::pair{32,
              std::map{
                std::pair{Operand::Bit, "r"},
                std::pair{Operand::Unsigned, "r"},
                std::pair{Operand::Signed, "r"},
                std::pair{Operand::Floating, "f"},
              }},
    std::pair{64,
              std::map{
                std::pair{Operand::Bit, "l"},
                std::pair{Operand::Unsigned, "l"},
                std::pair{Operand::Signed, "l"},
                std::pair{Operand::Floating, "d"},
              }},
    std::pair{128,
              std::map{
                std::pair{Operand::Bit, "l"},
                std::pair{Operand::Unsigned, "l"},
                std::pair{Operand::Signed, "l"},
                std::pair{Operand::Floating, "d"},
              }},
  };

  if (sz == 16)
  {
    return {"h"};
  }
  else
  {
    return constraint_map[sz][op];
  }
}

enum class Semantic
{
  Relaxed,
  Release,
  Acquire,
  Acq_Rel,
  Seq_Cst,
  Volatile,
};

inline std::string semantic(Semantic sem)
{
  static std::map sem_map = {
    std::pair{Semantic::Relaxed, ".relaxed"},
    std::pair{Semantic::Release, ".release"},
    std::pair{Semantic::Acquire, ".acquire"},
    std::pair{Semantic::Acq_Rel, ".acq_rel"},
    std::pair{Semantic::Seq_Cst, ".sc"},
    std::pair{Semantic::Volatile, ""},
  };
  return sem_map[sem];
}

inline std::string semantic_tag(Semantic sem)
{
  static std::map sem_map = {
    std::pair{Semantic::Relaxed, "__atomic_cuda_relaxed"},
    std::pair{Semantic::Release, "__atomic_cuda_release"},
    std::pair{Semantic::Acquire, "__atomic_cuda_acquire"},
    std::pair{Semantic::Acq_Rel, "__atomic_cuda_acq_rel"},
    std::pair{Semantic::Seq_Cst, "__atomic_cuda_seq_cst"},
    std::pair{Semantic::Volatile, "__atomic_cuda_volatile"},
  };
  return sem_map[sem];
}

enum class Scope
{
  Thread,
  Warp,
  CTA,
  Cluster,
  GPU,
  System,
};

inline std::string scope(Scope sco)
{
  static std::map sco_map = {
    std::pair{Scope::Thread, ""},
    std::pair{Scope::Warp, ""},
    std::pair{Scope::CTA, ".cta"},
    std::pair{Scope::Cluster, ".cluster"},
    std::pair{Scope::GPU, ".gpu"},
    std::pair{Scope::System, ".sys"},
  };
  return sco_map[sco];
}

inline std::string scope_tag(Scope sco)
{
  static std::map sco_map = {
    std::pair{Scope::Thread, "__thread_scope_thread_tag"},
    std::pair{Scope::Warp, ""},
    std::pair{Scope::CTA, "__thread_scope_block_tag"},
    std::pair{Scope::Cluster, "__thread_scope_cluster_tag"},
    std::pair{Scope::GPU, "__thread_scope_device_tag"},
    std::pair{Scope::System, "__thread_scope_system_tag"},
  };
  return sco_map[sco];
}

#endif // DEFINITIONS_H
