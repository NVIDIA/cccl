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
#  include "../traits.h"
#  include "cccl/c/types.h"
#  include "util/types.h"
#endif

template <typename ValueTp>
struct cccl_iterator_t_mapping
{
  bool is_pointer                             = false;
  int size                                    = 1;
  int alignment                               = 1;
  void (*advance)(void*, cuda::std::uint64_t) = nullptr;
  void (*dereference)(const void*, ValueTp*)  = nullptr;
  void (*assign)(const void*, ValueTp);

  using ValueT = ValueTp;
};

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
struct output_iterator_traits;

template <>
struct parameter_mapping<cccl_iterator_t>
{
  static const constexpr auto archetype = cccl_iterator_t_mapping<int>{};

  template <typename Traits>
  static std::string map(template_id<Traits>, cccl_iterator_t arg)
  {
    if (arg.advance.type != cccl_op_kind_t::CCCL_STATEFUL && arg.advance.type != cccl_op_kind_t::CCCL_STATELESS)
    {
      throw std::runtime_error("c.parallel: well-known operations are not allowed as an iterator's advance operation");
    }
    if (arg.dereference.type != cccl_op_kind_t::CCCL_STATEFUL && arg.dereference.type != cccl_op_kind_t::CCCL_STATELESS)
    {
      throw std::runtime_error("c.parallel: well-known operations are not allowed as an iterator's dereference "
                               "operation");
    }

    return std::format(
      "cccl_iterator_t_mapping<{}>{{.is_pointer = {}, .size = {}, .alignment = {}, .advance = {}, .{} = {}}}",
      cccl_type_enum_to_name(arg.value_type.type),
      arg.type == cccl_iterator_kind_t::CCCL_POINTER,
      arg.size,
      arg.alignment,
      arg.advance.name,
      std::is_same_v<Traits, output_iterator_traits> ? "assign" : "dereference",
      arg.dereference.name);
  }

  template <typename Traits>
  static std::string aux(template_id<Traits>, cccl_iterator_t arg)
  {
    if constexpr (std::is_same_v<Traits, output_iterator_traits>)
    {
      return std::format(
        R"output(
extern "C" __device__ void {0}(void *, {1});
extern "C" __device__ void {2}(const void *, {3});
)output",
        arg.advance.name,
        cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64),
        arg.dereference.name,
        cccl_type_enum_to_name(arg.value_type.type));
    }

    return std::format(
      R"input(
extern "C" __device__ void {0}(void *, {1});
extern "C" __device__ void {2}(const void *, {3}*);
)input",
      arg.advance.name,
      cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64),
      arg.dereference.name,
      cccl_type_enum_to_name(arg.value_type.type));
  }
};
#endif
