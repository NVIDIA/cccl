// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This benchmark only serves to check that cub::DeviceTransform correctly stores several independent output streams.
// It is not intended for tuning.

#include <thrust/device_vector.h>

#include <cuda/std/tuple>

#include "common.h"

using T = std::int8_t;

static constexpr auto startA      = T{11};
static constexpr auto startB      = T{2};
static constexpr auto startC      = T{1};
static constexpr auto startScalar = T{-2};

// multiple outputs and a function returning a tuple, like the primal projection step in cuOpt's
// PDHG solver (https://github.com/NVIDIA/cccl/issues/6143). 3 reads + 3 writes.
static void projection_zip_output(nvbench::state& state)
{
  const auto n = state.get_int64("Elements{io}");

  thrust::device_vector<T> x(n, startA);
  thrust::device_vector<T> obj(n, startB);
  thrust::device_vector<T> aty(n, startC);
  thrust::device_vector<T> next(n);
  thrust::device_vector<T> delta(n);
  thrust::device_vector<T> extrapolated(n);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n);
  state.add_global_memory_writes<T>(3 * n);

  const T step = startScalar;
  bench_transform(
    state,
    cuda::std::tuple{x.begin(), obj.begin(), aty.begin()},
    cuda::std::tuple{next.begin(), delta.begin(), extrapolated.begin()},
    n,
    [=] _CCCL_DEVICE(T xi, T oi, T ai) -> cuda::std::tuple<T, T, T> {
      const T next_xi = static_cast<T>(xi - (step * (oi - ai)));
      return {next_xi, next_xi - xi, next_xi - xi + next_xi};
    });
}

NVBENCH_BENCH(projection_zip_output).set_name("projection_zip_output").add_int64_power_of_two_axis("Elements{io}", {28});
