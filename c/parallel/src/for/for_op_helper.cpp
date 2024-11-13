//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

#include <cccl/c/types.h>
#include <for/for_op_helper.h>
#include <util/types.h>

static std::string get_for_kernel_iterator(cccl_iterator_t iter)
{
  const auto input_it_value_t = cccl_type_enum_to_string(iter.value_type.type);
  const auto offset_t         = cccl_type_enum_to_string(cccl_type_enum::UINT64);

  constexpr std::string_view stateful_iterator =
    R"XXX(
extern "C" __device__ {3} {4}(const void *self_ptr);
extern "C" __device__ void {5}(void *self_ptr, {0} offset);
struct __align__({1}) input_iterator_state_t {{;
  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type = {3};
  using difference_type = {0};
  using pointer = {3}*;
  using reference = {3}&;
  __device__ inline value_type operator*() const {{ return {4}(this); }}
  __device__ inline input_iterator_state_t& operator+=(difference_type diff) {{
      {5}(this, diff);
      return *this;
  }}
  __device__ inline value_type operator[](difference_type diff) const {{
      return *(*this + diff);
  }}
  __device__ inline input_iterator_state_t operator+(difference_type diff) const {{
      input_iterator_state_t result = *this;
      result += diff;
      return result;
  }}
  char data[{2}];
}};

using for_each_iterator_t = input_iterator_state_t;
)XXX";

  constexpr std::string_view stateless_iterator =
    R"XXX(
  using for_each_iterator_t = {0}*;
)XXX";

  return (iter.type == cccl_iterator_kind_t::iterator)
         ? std::format(
             stateful_iterator,
             offset_t, // 0 - type
             iter.alignment, // 1 - iter alignment
             iter.size, // 2 - iter size
             input_it_value_t, // 3 - iter value type
             iter.dereference.name, // 4 - deref
             iter.advance.name // 5 - advance name
             )
         : std::format(stateless_iterator, input_it_value_t);
}

static std::string get_for_kernel_user_op(cccl_op_t user_op, cccl_iterator_t iter)
{
  auto value_t = cccl_type_enum_to_string(iter.value_type.type);

  constexpr std::string_view op_format =
    R"XXX(
#if {0}
#  define _STATEFUL_USER_OP
#endif

#define _USER_OP {1}
#define _USER_OP_INPUT_T {2}

#if defined(_STATEFUL_USER_OP)
extern "C" __device__ void _USER_OP(void*, _USER_OP_INPUT_T*);
#else
extern "C" __device__ void _USER_OP(_USER_OP_INPUT_T*);
#endif

#if defined(_STATEFUL_USER_OP)
struct __align__({3}) user_op_t {{
  char data[{4}];
#else
struct user_op_t {{
#endif

  __device__ void operator()(_USER_OP_INPUT_T* input) {{
#if defined(_STATEFUL_USER_OP)
    _USER_OP(&data, input);
#else
    _USER_OP(input);
#endif
  }}
}};
)XXX";

  bool user_op_stateful = cccl_op_kind_t::stateful == user_op.type;

  return std::format(
    op_format,
    user_op_stateful, // 0 - stateful user op
    user_op.name, // 1 - user op function name
    value_t, // 2 - user op input type
    user_op.alignment, // 3 - state alignment
    user_op.size // 4 - state size
  );
}

std::string get_for_kernel(cccl_op_t user_op, cccl_iterator_t iter)
{
  auto storage_align = iter.value_type.alignment;
  auto storage_size  = iter.value_type.size;

  return std::format(
    R"XXX(
#include <cuda/std/iterator>
#include <cub/agent/agent_for.cuh>
#include <cub/device/dispatch/kernels/for_each.cuh>

struct __align__({2}) storage_t {{
  char data[{3}];
}};

// Iterator wrapper
{0}

// User operator wrapper
{1}

struct for_each_wrapper
{{
  for_each_iterator_t iterator;
  user_op_t user_op;

  __device__ void operator()(unsigned long long idx)
  {{
    user_op(iterator + idx);
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
    get_for_kernel_iterator(iter), // 0 - Iterator definition
    get_for_kernel_user_op(user_op, iter), // 1 - User op wrapper definition,
    storage_align, // 2 - User datatype alignment
    storage_size // 3 - User datatype size
  );
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
  size_t iter_size     = (cccl_iterator_kind_t::iterator == iterator.type) ? iterator.size : sizeof(void*);
  void* iterator_state = (cccl_iterator_kind_t::iterator == iterator.type) ? iterator.state : &iterator.state;

  // Do we need to valid user input? Alignments larger than the provided size?
  size_t user_size  = (cccl_op_kind_t::stateful == op.type) ? op.size : 0;
  size_t user_align = (cccl_op_kind_t::stateful == op.type) ? op.alignment : 0;

  auto [min_size, user_op_offset] = calculate_kernel_state_sizes(iter_size, user_size, user_align);

  for_each_default local_buffer{};
  char* iter_start = (char*) &local_buffer;

  // Check if local blueprint provides enough space
  bool use_allocated_storage = sizeof(for_each_default) < min_size;

  if (use_allocated_storage)
  {
    // Allocate required space
    iter_start = new char[min_size];
  }

  // Memcpy into either local or allocated buffer
  memcpy(iter_start, iterator_state, iter_size);
  if (cccl_op_kind_t::stateful == op.type)
  {
    char* user_start = iter_start + user_op_offset;
    memcpy(user_start, op.state, user_size);
  }

  // Return either local buffer or unique_ptr
  if (use_allocated_storage)
  {
    return for_each_kernel_state{std::unique_ptr<char[]>{iter_start}, user_op_offset};
  }
  else
  {
    return for_each_kernel_state{local_buffer, user_op_offset};
  }
}

void* for_each_kernel_state::get()
{
  return std::visit(
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
