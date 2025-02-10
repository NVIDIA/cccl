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

#include <cccl/c/types.h>

struct cccl_device_scan_build_result_t
{
  int cc;
  void* cubin;
  size_t cubin_size;
  CUlibrary library;
  cccl_type_info accumulator_type;
  CUkernel init_kernel;
  CUkernel scan_kernel;
  void* tile_state;
};

extern "C" CCCL_C_API CUresult cccl_device_scan_build(
  cccl_device_scan_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path) noexcept;

extern "C" CCCL_C_API CUresult cccl_device_scan(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  unsigned long long num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream) noexcept;

extern "C" CCCL_C_API CUresult cccl_device_scan_cleanup(cccl_device_scan_build_result_t* bld_ptr) noexcept;
