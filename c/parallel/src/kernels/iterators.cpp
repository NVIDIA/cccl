//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <format>

#include "cccl/c/types.h"
#include <kernels/iterators.h>
#include <util/errors.h>
#include <util/types.h>

constexpr std::string_view format_template = R"XXX(
#define DIFF_T {0}
#define OP_ALIGNMENT {1}
#define OP_SIZE {2}
#define VALUE_T {3}
#define DEREF {4}
#define ADVANCE {5}

// Kernel Source
{6}

#undef DIFF_T
#undef OP_ALIGNMENT
#undef OP_SIZE
#undef VALUE_T
#undef DEREF
#undef ADVANCE
)XXX";

std::string make_kernel_input_iterator(
  std::string_view diff_t,
  size_t alignment,
  size_t size,
  std::string_view iterator_name,
  std::string_view value_t,
  std::string_view deref,
  std::string_view advance)
{
  const std::string iter_def = std::format(R"XXX(
extern "C" __device__ void DEREF(const void *self_ptr, VALUE_T* result);
extern "C" __device__ void ADVANCE(void *self_ptr, DIFF_T offset);
struct __align__(OP_ALIGNMENT) {0} {{
  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type = VALUE_T;
  using difference_type = DIFF_T;
  using pointer = VALUE_T*;
  using reference = VALUE_T&;
  __device__ inline value_type operator*() const {{
    value_type result;
    DEREF(data, &result);
    return result;
  }}
  __device__ inline {0}& operator+=(difference_type diff) {{
      ADVANCE(data, diff);
      return *this;
  }}
  __device__ inline value_type operator[](difference_type diff) const {{
      return *(*this + diff);
  }}
  __device__ inline {0} operator+(difference_type diff) const {{
      {0} result = *this;
      result += diff;
      return result;
  }}
  char data[OP_SIZE];
}};
)XXX",
                                           iterator_name);

  return std::format(format_template, diff_t, alignment, size, value_t, deref, advance, iter_def);
};

std::string make_kernel_input_iterator(
  std::string_view offset_t, std::string_view iterator_name, std::string_view input_value_t, cccl_iterator_t iter)
{
  if (iter.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    return {};
  }

  return make_kernel_input_iterator(
    offset_t, iter.alignment, iter.size, iterator_name, input_value_t, iter.dereference.name, iter.advance.name);
}

std::string make_kernel_output_iterator(
  std::string_view diff_t,
  size_t alignment,
  size_t size,
  std::string_view iterator_name,
  std::string_view value_t,
  std::string_view deref,
  std::string_view advance)
{
  const std::string iter_def = std::format(R"XXX(
extern "C" __device__ void DEREF(const void *self_ptr, VALUE_T x);
extern "C" __device__ void ADVANCE(void *self_ptr, DIFF_T offset);
struct __align__(OP_ALIGNMENT) {0}_state_t {{
  char data[OP_SIZE];
}};
struct {0}_proxy_t {{
  __device__ {0}_proxy_t operator=(VALUE_T x) {{
    DEREF(&state, x);
    return *this;
  }}
  {0}_state_t state;
}};
struct {0} {{
  using iterator_category = cuda::std::random_access_iterator_tag;
  using difference_type   = DIFF_T;
  using value_type        = void;
  using pointer           = {0}_proxy_t*;
  using reference         = {0}_proxy_t;
  __device__ {0}_proxy_t operator*() const {{ return {{state}}; }}
  __device__ {0}& operator+=(difference_type diff) {{
      ADVANCE(&state, diff);
      return *this;
  }}
  __device__ {0}_proxy_t operator[](difference_type diff) const {{
    {0} result = *this;
    result += diff;
    return {{ result.state }};
  }}
  __device__ {0} operator+(difference_type diff) const {{
    {0} result = *this;
    result += diff;
    return result;
  }}
  {0}_state_t state;
}};
)XXX",
                                           iterator_name);

  return std::format(format_template, diff_t, alignment, size, value_t, deref, advance, iter_def);
};

std::string make_kernel_output_iterator(
  std::string_view offset_t, std::string_view iterator_name, std::string_view input_value_t, cccl_iterator_t iter)
{
  if (iter.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    return {};
  }

  return make_kernel_output_iterator(
    offset_t, iter.alignment, iter.size, iterator_name, input_value_t, iter.dereference.name, iter.advance.name);
}

std::string make_kernel_inout_iterator(
  std::string_view diff_t,
  size_t alignment,
  size_t size,
  std::string_view value_t,
  std::string_view deref,
  std::string_view advance)
{
  constexpr std::string_view format_template = R"XXX(
extern "C" __device__ {1}* {2}(const void *self_ptr);
extern "C" __device__ void {3}(void *self_ptr, {0} offset);

struct __align__({5}) output_iterator_state_t{{
  char data[{4}];
}};

struct output_iterator_t {{
  using iterator_category = cuda::std::random_access_iterator_tag;
  using difference_type   = {0};
  using value_type        = VALUE_T;
  using pointer           = output_iterator_proxy_t*;
  using reference         = output_iterator_proxy_t;
  __device__ {1} operator*() const {{ return {2}(&state); }}
  __device__ output_iterator_t& operator+=(difference_type diff) {{
      {3}(&state, diff);
      return *this;
  }}
  __device__ output_iterator_proxy_t operator[](difference_type diff) const {{
    output_iterator_t result = *this;
    result += diff;
    return {{ result.state }};
  }}
  __device__ output_iterator_t operator+(difference_type diff) const {{
    output_iterator_t result = *this;
    result += diff;
    return result;
  }}
  output_iterator_state_t state;
}};
)XXX";

  return std::format(format_template, diff_t, alignment, size, value_t, deref, advance);
};

std::string make_kernel_inout_iterator(std::string_view offset_t, std::string_view input_value_t, cccl_iterator_t iter)
{
  if (iter.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    return {};
  }

  return make_kernel_inout_iterator(
    offset_t, iter.alignment, iter.size, input_value_t, iter.dereference.name, iter.advance.name);
}
