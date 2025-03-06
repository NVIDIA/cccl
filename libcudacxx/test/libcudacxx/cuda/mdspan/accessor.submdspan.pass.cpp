//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename Mdspan>
__host__ __device__ void test_submdspan()
{
  int array[] = {1, 2, 3, 4};
  Mdspan md{array, cuda::std::dims<1>{4}};
  auto submd = cuda::std::submdspan(md, cuda::std::pair{1, 3});
#if defined(__CUDA_ARCH__)
  if constexpr (cuda::is_device_accessible_v<Mdspan>)
  {
    assert(submd(0) == 2);
    assert(submd(1) == 3);
  }
#else
  if constexpr (cuda::is_host_accessible_v<Mdspan>)
  {
    assert(submd(0) == 2);
    assert(submd(1) == 3);
  }
#endif
  unused(submd);
}

int main(int, char**)
{
  test_submdspan<cuda::host_mdspan<int, cuda::std::dims<1>>>();
  test_submdspan<cuda::device_mdspan<int, cuda::std::dims<1>>>();
  test_submdspan<cuda::managed_mdspan<int, cuda::std::dims<1>>>();
  return 0;
}
