// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

int main()
{
  namespace stdexec = cuda::std::execution;

  int* ptr{};
  auto env = stdexec::env{cuda::execution::determinism::run_to_run};

  // expected-error {{"Determinism should be used inside requires to have an effect."}}
  cub::DeviceReduce::Reduce(ptr, ptr, 0, cuda::std::plus<>{}, 0, env);
}
