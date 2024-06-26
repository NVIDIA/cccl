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

static std::string constraints(Operand op, size_t sz)
{
  static std::map constraint_map = {
    std::pair{4,
              std::map{
                std::pair{Operand::Bit, "r"},
                std::pair{Operand::Unsigned, "r"},
                std::pair{Operand::Signed, "r"},
                std::pair{Operand::Floating, "f"},
              }},
    std::pair{8,
              std::map{
                std::pair{Operand::Bit, "l"},
                std::pair{Operand::Unsigned, "l"},
                std::pair{Operand::Signed, "l"},
                std::pair{Operand::Floating, "d"},
              }}};

  if (sz == 2)
  {
    return {"h"};
  }
  else
  {
    return constraint_map[sz][op];
  }
}

enum class Semantics
{
  Relaxed,
  Release,
  Acquire,
  Acq_Rel,
  Seq_Cst,
  Volatile,
};

static std::string semantics(Semantics sem)
{
  static std::map sem_map = {
    std::pair{Semantics::Relaxed, ".relaxed"},
    std::pair{Semantics::Release, ".release"},
    std::pair{Semantics::Acquire, ".acquire"},
    std::pair{Semantics::Acq_Rel, ".acq_rel"},
    std::pair{Semantics::Seq_Cst, ".sc"}, // Only used in seq_cst fence operations
    std::pair{Semantics::Volatile, ""},
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

#endif // DEFINITIONS_H
