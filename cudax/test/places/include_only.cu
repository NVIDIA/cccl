//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Single umbrella include: must pull in every public places surface (including
// partition strategies) without requiring clients to list internal headers.

#include <cuda/experimental/places.cuh>

using namespace cuda::experimental::places;

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
