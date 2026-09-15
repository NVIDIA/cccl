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
 * @brief N-ary zip_transform: one fused pass over co-partitioned sharded
 *        views (the multi-operand solver-update shape), in-place support,
 *        and the co-partitioning check.
 */

#include <cuda/experimental/sharded.cuh>

#include <stdexcept>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
// w*(c*reflected + (1-c)*current) + (1-w)*initial — the 3-input update shape
struct pdhg_like
{
  double w = 0.5, c = 1.0;
  __host__ __device__ double operator()(double reflected, double current, double initial) const
  {
    return w * (c * reflected + (1.0 - c) * current) + (1.0 - w) * initial;
  }
};
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));
  auto group     = place_group{make_locality_domain_grid()};
  const size_t n = 1000001;

  auto reflected = sharded_array<double>::allocate(group, n);
  auto current   = sharded_array<double>::allocate(group, n);
  auto initial   = sharded_array<double>::allocate(group, n);
  fill(reflected, 8.0);
  fill(current, 4.0);
  fill(initial, 2.0);

  // out == current (in-place into one of the inputs, as the solver does)
  zip_transform(current, pdhg_like{}, reflected, current, initial);
  // 0.5*(1*8 + 0*4) + 0.5*2 = 5
  EXPECT(reduce(current, ::cuda::std::plus<double>{}, 0.0) == 5.0 * n);

  // binary through the same entry point
  zip_transform(current, ::cuda::std::plus<double>{}, current, initial);
  EXPECT(reduce(current, ::cuda::std::plus<double>{}, 0.0) == 7.0 * n);

  // co-partitioning violation is refused
  auto other = sharded_array<double>::allocate(group, n + 1);
  bool threw = false;
  try
  {
    zip_transform(current, ::cuda::std::plus<double>{}, current, other);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);

  return 0;
}
