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
#include <vector>

#include <fmt/format.h>

enum class Operand
{
  Floating,
  Unsigned,
  Signed,
  Bit,
};

static std::string operand(Operand op)
{
  static std::map op_map = {
    std::pair{Operand::Floating, "f"},
    std::pair{Operand::Unsigned, "u"},
    std::pair{Operand::Signed, "s"},
    std::pair{Operand::Bit, "b"},
  };
  return op_map[op];
}

static std::string operand_deduced(Operand op)
{
  static std::map op_map = {
    std::pair{Operand::Floating, ""},
    std::pair{Operand::Unsigned, "u"},
    std::pair{Operand::Signed, "s"},
    std::pair{Operand::Bit, "b"},
  };
  return op_map[op];
}

static std::string operand_proxy_type(Operand op, size_t sz)
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

static std::string constraints(Operand op, size_t sz)
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
              }}};

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

static std::string semantic(Semantic sem)
{
  static std::map sem_map = {
    std::pair{Semantic::Relaxed, ".relaxed"},
    std::pair{Semantic::Release, ".release"},
    std::pair{Semantic::Acquire, ".acquire"},
    std::pair{Semantic::Acq_Rel, ".acq_rel"},
    std::pair{Semantic::Seq_Cst, ".sc"}, // Only used in seq_cst fence operations
    std::pair{Semantic::Volatile, ""},
  };
  return sem_map[sem];
}

static std::string semantic_tag(Semantic sem)
{
  static std::map sem_map = {
    std::pair{Semantic::Relaxed, "relaxed"},
    std::pair{Semantic::Release, "release"},
    std::pair{Semantic::Acquire, "acquire"},
    std::pair{Semantic::Acq_Rel, "acq_rel"},
    std::pair{Semantic::Seq_Cst, "seq_cst"}, // Only used in seq_cst fence operations
    std::pair{Semantic::Volatile, "volatile"},
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

static std::string scope(Scope sco)
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

static std::string scope_tag(Scope sco)
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
