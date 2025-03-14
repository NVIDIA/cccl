//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <cuda.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

typedef struct cccl_device_radix_sort_build_result_t
{
  int cc;
  void* cubin;
  size_t cubin_size;
  CUlibrary library;
  CUkernel single_tile_kernel;
  CUkernel upsweep_kernel;
  CUkernel alt_upsweep_kernel;
  CUkernel scan_bins_kernel;
  CUkernel downsweep_kernel;
  CUkernel alt_downsweep_kernel;
  CUkernel histogram_kernel;
  CUkernel exclusive_sum_kernel;
  CUkernel onesweep_kernel;
} cccl_device_radix_sort_build_result_t;
