//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Correctness of the elementwise sharded algorithms (fill, sequence,
 *        iota, tabulate, generate, for_each, transform) against host
 *        references, over multiple places.
 */

#include <cuda/experimental/sharded.cuh>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct times_two_plus_index
{
  __host__ __device__ long long operator()(size_t i) const
  {
    return 2 * static_cast<long long>(i) + 7;
  }
};

struct negate_op
{
  __host__ __device__ long long operator()(long long x) const
  {
    return -x;
  }
};

struct saxpy_op
{
  __host__ __device__ long long operator()(long long x, long long y) const
  {
    return 3 * x + y;
  }
};

struct set_to_index
{
  __host__ __device__ void operator()(long long& v, size_t i) const
  {
    v += static_cast<long long>(i);
  }
};

struct const_gen
{
  __host__ __device__ long long operator()() const
  {
    return 42;
  }
};

void test_fill_and_sequence(place_group& group)
{
  const size_t n = 100003;
  auto data      = sharded_array<long long>::allocate(group, n);

  fill(data, 17LL);
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 17LL);
  }

  sequence(data, default_envs(data), 5LL, 3LL); // 5, 8, 11, ...
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 5LL + 3LL * static_cast<long long>(i));
  }

  iota(data, 100LL);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 100LL + static_cast<long long>(i));
  }
}

void test_tabulate_generate_for_each(place_group& group)
{
  const size_t n = 65537;
  auto data      = sharded_array<long long>::allocate(group, n);

  tabulate(data, times_two_plus_index{});
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 2 * static_cast<long long>(i) + 7);
  }

  generate(data, const_gen{});
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 42LL);
  }

  // for_each sees the GLOBAL index: 42 + i
  for_each(data, set_to_index{});
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 42LL + static_cast<long long>(i));
  }
}

void test_transform(place_group& group)
{
  const size_t n = 50000;
  auto a         = sharded_array<long long>::allocate(group, n);
  iota(a, 0LL);

  // In-place
  transform(a, negate_op{});
  ::std::vector<long long> host(n);
  a.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == -static_cast<long long>(i));
  }

  // Unary out-of-place
  auto b = sharded_array<long long>::allocate_like(a);
  zip_transform(b, negate_op{}, a);
  b.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i));
  }

  // Binary: c = 3*a + b = -3i + i = -2i
  auto c = sharded_array<long long>::allocate_like(a);
  zip_transform(c, saxpy_op{}, a, b);
  c.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == -2 * static_cast<long long>(i));
  }

  // Incompatible layouts must throw
  auto other = sharded_array<long long>::allocate({{n / 2, data_place::device(0), exec_place::device(0), nullptr}});
  bool threw = false;
  try
  {
    zip_transform(other, negate_op{}, a);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  test_fill_and_sequence(group);
  test_tabulate_generate_for_each(group);
  test_transform(group);

  return 0;
}
