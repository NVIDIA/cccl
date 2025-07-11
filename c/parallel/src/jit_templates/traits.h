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

#include <cuda/std/type_traits>

#include <format>
#include <string>

#include "../util/errors.h"

extern const char* jit_template_header_contents;

template <typename Arg>
struct parameter_mapping;

template <typename Tpl>
struct template_id
{};

struct specialization
{
  std::string type_name;
  std::string aux_code;
};

template <typename Tag,
          typename Traits,
          typename... Args,
          typename = Traits::template type<void, parameter_mapping<Args>::archetype...>>
specialization get_specialization(template_id<Traits> id, Args... args)
{
  if constexpr (requires { Traits::template special<Tag>(args...); })
  {
    if (auto result = Traits::template special<Tag>(args...))
    {
      return *result;
    }
  }

  std::string tag_name;
  check(nvrtcGetTypeName<Tag>(&tag_name));

  return {std::format("{}<{}{}>", Traits::name, tag_name, ((", " + parameter_mapping<Args>::map(id, args)) + ...)),
          std::format("struct {};", tag_name) + (parameter_mapping<Args>::aux(id, args) + ...)};
}
