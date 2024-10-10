//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

void rec_func(exec_place_grid places)
{
  if (places.size() == 1)
  {
    // places->print("SINGLE");
  }
  else
  {
    // places->print("REC");
    for (int i = 0; i < 2; i++)
    {
      // Take every other places from the grid
      auto half_places = partition_cyclic(places, dim4(2), pos4(i));
      rec_func(half_places);
    }
  }
}

int main()
{
  auto places = exec_place::all_devices();
  // places->print("ALL");

  rec_func(places);

  return 0;
}
