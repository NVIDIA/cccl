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

template <typename T>
struct cccl_type_info_mapping
{
  using Type = T;
};

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
#  include "../traits.h"

template <>
struct parameter_mapping<cccl_type_info>
{
  static const constexpr auto archetype = cccl_type_info_mapping<int>{};

  template <typename TplId>
  static std::string map(TplId, cccl_type_info arg)
  {
    return std::format("cccl_type_info_mapping<{}>{{}}", cccl_type_enum_to_name(arg.type));
  }

  template <typename TplId>
  static std::string aux(TplId, cccl_type_info)
  {
    return {};
  }
};
#endif
