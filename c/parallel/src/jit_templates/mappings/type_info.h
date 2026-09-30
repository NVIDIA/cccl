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

  template <typename TplId, typename ArgT>
  static std::string map(TplId, ArgT arg)
  {
    using traits       = arg_traits<cuda::std::decay_t<ArgT>>;
    using storage_type = typename traits::storage_type;
    const auto& value  = traits::unwrap(arg);
    return std::format("cccl_type_info_mapping<{}>{{}}", cccl_type_enum_to_name<storage_type>(value.type));
  }

  template <typename TplId, typename ArgT>
  static std::string aux(TplId, ArgT)
  {
    return {};
  }
};
#endif
