//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
#  include <cuda/std/optional>
#  include <cuda/std/span>

#  include <stdexcept>

#  include "../traits.h"
#  include <cccl/c/types.h>
#endif

template <auto Operation = nullptr>
struct cccl_op_t_mapping
{
  bool is_stateless               = false;
  int size                        = 1;
  int alignment                   = 1;
  static constexpr auto operation = Operation;
};

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
template <>
struct parameter_mapping<cccl_op_t>
{
  static const constexpr auto archetype = cccl_op_t_mapping<>{};

  template <typename Traits, typename ArgT>
  static std::string map(template_id<Traits>, ArgT arg)
  {
    const auto& value = arg_traits<cuda::std::decay_t<ArgT>>::unwrap(arg);
    // This mapping requires a function name, even if the implementation is linked later.
    // Reject missing names here so callers such as Python get a meaningful error.
    if (value.name == nullptr || value.name[0] == '\0')
    {
      if (value.type != cccl_op_kind_t::CCCL_STATELESS && value.type != cccl_op_kind_t::CCCL_STATEFUL)
      {
        throw ::std::invalid_argument(
          "c.parallel: built-in operations are not supported for storage types (including structs) without a custom "
          "operation implementation.");
      }
      throw ::std::invalid_argument("c.parallel: a custom operation requires a non-empty function name.");
    }
    return std::format(
      "cccl_op_t_mapping<{}>{{.is_stateless = {}, .size = {}, .alignment = {}}}",
      value.name,
      value.type != cccl_op_kind_t::CCCL_STATEFUL,
      value.size,
      value.alignment);
  }

  template <typename Traits, typename ArgT>
  static std::string aux(template_id<Traits>, ArgT arg)
  {
    const auto& value = arg_traits<cuda::std::decay_t<ArgT>>::unwrap(arg);

    std::string args;
    if (value.type == cccl_op_kind_t::CCCL_STATEFUL)
    {
      args += "void*";
      if constexpr (Traits::arity > 0)
      {
        args += ", ";
      }
    }

    for (int i = 0; i < Traits::arity; ++i)
    {
      args += "const void*";
      args += ", ";
    }

    args += "void*";

    return std::format(
      R"(
        extern "C" __device__ void {}({});
        )",
      value.name,
      args);
  }
};
#endif
