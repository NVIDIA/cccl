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
  bool is_pointer                         = false;
  int size                                = 1;
  int alignment                           = 1;
  void (*advance)(void*, const void*)     = nullptr;
  void (*dereference)(const void*, void*) = nullptr;
  void (*assign)(void*, const void*)      = nullptr;

  using ValueT = ValueTp;
};

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
struct output_iterator_traits;

template <>
struct parameter_mapping<cccl_iterator_t>
{
  static const constexpr auto archetype = cccl_iterator_t_mapping<int>{};

  template <typename Traits, typename ArgT>
  static std::string map(template_id<Traits>, ArgT arg)
  {
    using traits       = arg_traits<cuda::std::decay_t<ArgT>>;
    using storage_type = typename traits::storage_type;
    const auto& value  = traits::unwrap(arg);

    if (value.advance.type != cccl_op_kind_t::CCCL_STATEFUL && value.advance.type != cccl_op_kind_t::CCCL_STATELESS)
    {
      throw std::runtime_error("c.parallel: well-known operations are not allowed as an iterator's advance operation");
    }
    if (value.dereference.type != cccl_op_kind_t::CCCL_STATEFUL
        && value.dereference.type != cccl_op_kind_t::CCCL_STATELESS)
    {
      throw std::runtime_error("c.parallel: well-known operations are not allowed as an iterator's dereference "
                               "operation");
    }

    return std::format(
      "cccl_iterator_t_mapping<{}>{{.is_pointer = {}, .size = {}, .alignment = {}, .advance = {}, .{} = {}}}",
      cccl_type_enum_to_name<storage_type>(value.value_type.type),
      value.type == cccl_iterator_kind_t::CCCL_POINTER,
      value.size,
      value.alignment,
      value.advance.name,
      std::is_same_v<Traits, output_iterator_traits> ? "assign" : "dereference",
      value.dereference.name);
  }

  template <typename Traits, typename ArgT>
  static std::string aux(template_id<Traits>, ArgT arg)
  {
    using traits       = arg_traits<cuda::std::decay_t<ArgT>>;
    using storage_type = typename traits::storage_type;
    const auto& value  = traits::unwrap(arg);

    if constexpr (std::is_same_v<Traits, output_iterator_traits>)
    {
      return std::format(
        R"output(
extern "C" __device__ void {0}(void *, const void*);
extern "C" __device__ void {1}(void *, const void*);
)output",
        value.advance.name,
        value.dereference.name);
    }

    return std::format(
      R"input(
extern "C" __device__ void {0}(void *, const void*);
extern "C" __device__ void {1}(const void *, void*);
)input",
      value.advance.name,
      value.dereference.name);
  }
};
#endif
