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
#include "../util/types.h"

extern const char* jit_template_header_contents;

template <typename Arg>
struct parameter_mapping;

template <typename Tpl>
struct template_id
{};

// tagged_arg is needed to pass storage type information to the parameter
// mapping. This is needed because different args may have different storage
// types.
template <typename StorageT, typename T>
struct tagged_arg
{
  using storage_type = StorageT;
  using value_type   = T;
  T value;
};

template <typename T>
struct is_tagged_arg : cuda::std::false_type
{};

template <typename StorageT, typename T>
struct is_tagged_arg<tagged_arg<StorageT, T>> : cuda::std::true_type
{};

template <typename T>
struct arg_traits
{
  using storage_type = storage_t;
  using value_type   = T;

  static const T& unwrap(const T& value)
  {
    return value;
  }
};

template <typename StorageT, typename T>
struct arg_traits<tagged_arg<StorageT, T>>
{
  using storage_type = StorageT;
  using value_type   = T;

  static const T& unwrap(const tagged_arg<StorageT, T>& value)
  {
    return value.value;
  }
};

template <typename T>
struct mapping_arg_type
{
  using type = T;
};

template <typename StorageT, typename T>
struct mapping_arg_type<tagged_arg<StorageT, T>>
{
  using type = T;
};

struct specialization
{
  std::string type_name;
  std::string aux_code = "";
};

template <typename Tag, typename Traits, typename... Args>
specialization get_specialization(template_id<Traits> id, Args... args)
{
#ifdef __CUDA_ARCH__
  return specialization{};
#else
  if constexpr (requires { Traits::template special<Tag>(args...); })
  {
    if (auto result = Traits::template special<Tag>(args...))
    {
      return *result;
    }
  }

  std::string tag_name;
  check(cccl_type_name_from_nvrtc<Tag>(&tag_name));

  auto map = [&](auto arg) {
    using arg_t = cuda::std::decay_t<decltype(arg)>;
    using map_t = typename mapping_arg_type<arg_t>::type;
    return parameter_mapping<map_t>::map(id, arg);
  };

  auto aux = [&](auto arg) {
    using arg_t = cuda::std::decay_t<decltype(arg)>;
    using map_t = typename mapping_arg_type<arg_t>::type;
    return parameter_mapping<map_t>::aux(id, arg);
  };

  return {std::format("{}<{}{}>", Traits::name, tag_name, ((", " + map(args)) + ...)),
          std::format("struct {};", tag_name) + (aux(args) + ...)};
#endif
}
