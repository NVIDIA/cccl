//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdlib>
#include <format>
#include <memory>
#include <type_traits>

#include "cccl/c/types.h"
#include <for/for_op_helper.h>
#include <util/small_storage.h>

std::string get_for_kernel_op_declaration(cccl_op_t op, std::string state_identifier)
{
  return (op.type == cccl_op_kind_t::stateful)
         ? std::format(R"XXX(
extern "C" __device__ void {0}(void* state, void* data);
struct __align__({1}) {3} {{
  char state[{2}];
}};
)XXX",
                       op.name, // 0 - op name
                       op.alignment, // 1 - op alignment
                       op.size, // 2 - op size
                       state_identifier // 3 - storage type identifier
                       )
         : std::format(R"XXX(
extern "C" __device__ void {0}(void* data);
struct {1}{{}};
)XXX",
                       op.name, // 0 - op name
                       state_identifier // 1 - storage type identifier
           );
}

std::string get_for_kernel_op_invocation(cccl_op_t op, std::string state_var)
{
  return (op.type == cccl_op_kind_t::stateful)
         ? std::format("{0}(&state, data + (idx * size));", op.name)
         : std::format("{0}(data + (idx * size));" op.name);
}

std::string get_for_kernel_iterator_declaration(cccl_iterator_t iter)
{
  return (iter.type == cccl_iterator_kind_t::iterator)
         ? std::format(
             R"XXX(
extern "C" __device__ void {0}(void*);
extern "C" __device__ void {1}(void*);
struct __align__({2}) for_kernel_iterator {{
  char state[{3}];
}};
)XXX",
             iter.advance.name, // 0 - advance
             iter.dereference.name, // 1 - deref
             iter.alignment, // 2 - iter alignment
             iter.size // 3 - iter size
             )
         : std::format(R"XXX(
extern "C" __device__ void {0}(void* data);
struct {1}{{}};
)XXX",
                       op.name, // 0 - op name
                       state_identifier // 1 - storage type identifier
           );
}

std::string get_for_kernel(cccl_op_t user_op, cccl_iterator_t iter)
{
    return std::format(
    R"XXX(
#include <cuda/std/iterator>
#include <cub/agent/agent_for.cuh>
#include <cub/device/dispatch/kernels/for_each.cuh>

struct for_each_wrapper
{{
  __device__ void operator()(size_t idx)
  {{

  }}
}}

struct op_iter_wrapper
{{
  __device__ void operator()(difference_type idx)
  {{
#if {5} // enable stateful op dispatch
    {4}(&state, data + (idx*size));
#else
    {4}(data + (idx*size));
#endif
  }}
}};

using policy_dim_t = cub::detail::for_each::policy_t<256, 2>;

struct device_for_policy
{{
  struct ActivePolicy
  {{
    using for_policy_t = policy_dim_t;
  }};
}};
)XXX",
}

constexpr static std::tuple<size_t, size_t>
calculate_kernel_state_sizes(size_t iter_size, size_t user_size, size_t user_align)
{
  size_t min_size       = iter_size;
  size_t user_op_offset = 0;

  if (user_size)
  {
    // Add space to match alignment provided by user
    size_t alignment = (min_size & (user_align - 1));
    if (alignment)
    {
      min_size += user_align - alignment;
    }
    // Capture offset where user function state begins
    user_op_offset = min_size;
    min_size += user_size;
  }

  return {min_size, user_op_offset};
}

static_assert(calculate_kernel_state_sizes(4, 8, 8) == std::tuple<size_t, size_t>{16, 8});
static_assert(calculate_kernel_state_sizes(2, 8, 8) == std::tuple<size_t, size_t>{16, 8});
static_assert(calculate_kernel_state_sizes(16, 8, 8) == std::tuple<size_t, size_t>{24, 16});
static_assert(calculate_kernel_state_sizes(8, 8, 8) == std::tuple<size_t, size_t>{16, 8});
static_assert(calculate_kernel_state_sizes(8, 16, 8) == std::tuple<size_t, size_t>{24, 8});
static_assert(calculate_kernel_state_sizes(8, 16, 16) == std::tuple<size_t, size_t>{32, 16});

for_each_kernel_state make_for_kernel_state(cccl_op_t op, cccl_iterator_t iterator)
{
  // Iterator is either a pointer or a stateful object, allocate space according to its size or alignment
  size_t iter_size     = (iterator.type == cccl_iterator_kind_t::iterator) ? iterator.size : sizeof(void*);
  size_t iter_align    = (iterator.type == cccl_iterator_kind_t::iterator) ? iterator.alignment : sizeof(void*);
  void* iterator_state = (iterator.type == cccl_iterator_kind_t::iterator) ? iterator.state : &iterator.state;

  // Do we need to valid user input? Alignments larger than the provided size?
  size_t user_size  = (op.type == cccl_op_kind_t::stateful) ? op.size : 0;
  size_t user_align = (op.type == cccl_op_kind_t::stateful) ? op.alignment : 0;

  auto [min_size, user_op_offset] = calculate_kernel_state_sizes(iter_size, user_size, user_align);

  for_each_default local_buffer{};
  char* iter_start = (char*) &local_buffer;

  // Check if local blueprint provides enough space
  if (sizeof(for_each_default) >= min_size)
  {
    // Allocate required space
    iter_start = (char*) malloc(min_size);
  }

  // Memcpy into either local or allocated buffer
  memcpy(iter_start, iterator_state, iter_size);
  if (op.type == cccl_op_kind_t::stateful)
  {
    char* user_start = iter_start + user_op_offset;
    memcpy(user_start, op.state, user_size);
  }

  // Return either local buffer or unique_ptr
  if (sizeof(for_each_default) >= min_size)
  {
    return for_each_kernel_state{local_buffer, user_op_offset};
  }
  else
  {
    return for_each_kernel_state{std::unique_ptr<void>(iter_start), user_op_offset};
  }
}

void* for_each_kernel_state::get()
{
  std::visit(
    [](auto&& v) -> void* {
      using state_t = std::decay_t<decltype(v)>;
      if constexpr (std::is_same_v<for_each_default, state_t>)
      {
        // Return the locally stored object as a void*
        return &v;
      }
      else
      {
        // Return the allocated space as a void*
        return v.get();
      }
    },
    for_each_arg);
}
