//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__places/data_place_impl.cuh>
#include <cuda/experimental/__places/data_place_interface.cuh>
#include <cuda/experimental/__places/exec/cuda_stream.cuh>
#include <cuda/experimental/__places/exec/green_context.cuh>
#include <cuda/experimental/__places/exec/green_ctx_view.cuh>
#include <cuda/experimental/__places/place_partition.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__places/stream_pool.cuh>

using namespace cuda::experimental::stf;

int main()
{
  auto host_place = data_place::host();
  auto dev0_place = data_place::device(0);
  auto exec_host  = exec_place::host();
  auto exec_dev0  = exec_place::device(0);

  (void) host_place;
  (void) dev0_place;
  (void) exec_host;
  (void) exec_dev0;

  return 0;
}
