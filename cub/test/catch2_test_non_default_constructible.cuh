// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/execution_policy.h>
#include <thrust/reverse.h>
#include <thrust/tabulate.h>

#include <cuda/buffer>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <ostream>

#include <c2h/device_and_stream.h>

// A trivially copyable type without a default constructor
struct not_default_constructible_t
{
  int value;

  __host__ __device__ explicit not_default_constructible_t(int v)
      : value(v)
  {}

  __host__ __device__ friend bool
  operator==(const not_default_constructible_t& lhs, const not_default_constructible_t& rhs)
  {
    return lhs.value == rhs.value;
  }

  __host__ __device__ friend bool
  operator!=(const not_default_constructible_t& lhs, const not_default_constructible_t& rhs)
  {
    return !(lhs == rhs);
  }

  __host__ __device__ friend bool
  operator<(const not_default_constructible_t& lhs, const not_default_constructible_t& rhs)
  {
    return lhs.value < rhs.value;
  }

  // for printing mismatches
  friend std::ostream& operator<<(std::ostream& os, const not_default_constructible_t& v)
  {
    return os << v.value;
  }
};

static_assert(!cuda::std::is_default_constructible_v<not_default_constructible_t>);
static_assert(cuda::std::is_trivially_copyable_v<not_default_constructible_t>);

struct make_not_default_constructible_op_t
{
  int stride;

  __host__ __device__ not_default_constructible_t operator()(int i) const
  {
    return not_default_constructible_t{i * stride};
  }
};

// Device storage for not_default_constructible_t, which thrust::device_vector cannot hold. Elements are constructed on
// the device. Holds 0, stride, 2 * stride, ..., (num_items - 1) * stride, or the same values in reverse order.
// The stream must outlive the returned buffer.
inline auto make_ndc_buffer(const cuda::stream& stream, int num_items, bool reversed = false, int stride = 1)
{
  auto buffer =
    cuda::make_device_buffer<not_default_constructible_t>(stream, c2h::current_device(), num_items, cuda::no_init);
  const auto policy = thrust::cuda::par.on(stream.get());
  thrust::tabulate(policy, buffer.data(), buffer.data() + num_items, make_not_default_constructible_op_t{stride});
  if (reversed)
  {
    thrust::reverse(policy, buffer.data(), buffer.data() + num_items);
  }
  stream.sync();
  return buffer;
}
