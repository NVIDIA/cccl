//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc

#include <cuda/__mdspan/host_device_mdspan.h>

#include "test_macros.h"

int main(int, char**)
{
  int array[] = {1, 2, 3, 4};
  using ext_t = cuda::std::extents<int, 4>;
  cuda::device_mdspan<int, ext_t> d_md{array, ext_t{}};
#if !defined(__CUDA_ARCH__)
  unused(d_md[0]);
#endif
  return 0;
}
