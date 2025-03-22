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
#  include <cuda/std/cstddef>
#  include <cuda/std/type_traits>

#  include <cccl/c/types.h>
#endif

#include "../mappings/operation.h"
#include "../mappings/type_info.h"

template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping RetT, cccl_type_info_mapping... ArgTs>
struct stateless_user_operation
{
  __device__ decltype(RetT)::Type operator()(decltype(ArgTs)::Type... args) const
  {
    return reinterpret_cast<decltype(RetT)::Type (*)(decltype(ArgTs)::Type...)>(Operation.operation)(
      std::move(args)...);
  }
};

template <cuda::std::size_t Size, cuda::std::size_t Alignment>
struct alignas(Alignment) user_operation_state
{
  char data[Size];
};

template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping RetT, cccl_type_info_mapping... ArgTs>
struct stateful_user_operation
{
  user_operation_state<Operation.size, Operation.alignment> state;
  __device__ decltype(RetT)::Type operator()(decltype(ArgTs)::Type... args)
  {
    return reinterpret_cast<decltype(RetT)::Type (*)(void*, decltype(ArgTs)::Type...)>(
      Operation.operation)(&state, std::move(args)...);
  }
};

struct user_operation_traits
{
  static const constexpr auto name = "user_operation_traits::type";
  template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping RetT, cccl_type_info_mapping... ArgTs>
  using type = cuda::std::conditional_t<Operation.is_stateless,
                                        stateless_user_operation<Tag, Operation, RetT, ArgTs...>,
                                        stateful_user_operation<Tag, Operation, RetT, ArgTs...>>;
};

struct binary_user_operation_traits
{
  static const constexpr auto name = "binary_user_operation_traits::type";
  template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping ValueT>
  using type = user_operation_traits::type<Tag, Operation, ValueT, ValueT, ValueT>;
};
