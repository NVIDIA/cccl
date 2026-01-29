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

#  include "../traits.h"
#  include <cccl/c/types.h>
#endif

struct cccl_op_t_mapping
{
  bool is_stateless   = false;
  int size            = 1;
  int alignment       = 1;
  void (*operation)() = nullptr;
};

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
template <>
struct parameter_mapping<cccl_op_t>
{
  static const constexpr auto archetype = cccl_op_t_mapping{};

  template <typename Traits, typename ArgT>
  static std::string map(template_id<Traits>, ArgT arg)
  {
    const auto& value = arg_traits<cuda::std::decay_t<ArgT>>::unwrap(arg);
    return std::format(
      "cccl_op_t_mapping{{.is_stateless = {}, .size = {}, .alignment = {}, .operation = {}}}",
      value.type != cccl_op_kind_t::CCCL_STATEFUL,
      value.size,
      value.alignment,
      value.name);
  }

  template <typename Traits, typename ArgT>
  static std::string aux(template_id<Traits>, ArgT arg)
  {
    const auto& value = arg_traits<cuda::std::decay_t<ArgT>>::unwrap(arg);
    return std::format(R"(
        extern "C" __device__ void {}();
        )",
                       value.name);
  }
};
#endif
