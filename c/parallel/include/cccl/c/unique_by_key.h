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
#include <stdint.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

typedef struct cccl_device_unique_by_key_build_result_t
{
  int cc;
  void* cubin;
  size_t cubin_size;
  CUlibrary library;
  CUkernel compact_init_kernel;
  CUkernel sweep_kernel;
  size_t description_bytes_per_tile;
  size_t payload_bytes_per_tile;
} cccl_device_unique_by_key_build_result_t;

CCCL_C_API CUresult cccl_device_unique_by_key_build(
  cccl_device_unique_by_key_build_result_t* build,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

// Extended version with build configuration
CCCL_C_API CUresult cccl_device_unique_by_key_build_ex(
  cccl_device_unique_by_key_build_result_t* build,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_unique_by_key(
  cccl_device_unique_by_key_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_out,
  cccl_iterator_t d_num_selected_out,
  cccl_op_t op,
  uint64_t num_items,
  CUstream stream);

CCCL_C_API CUresult cccl_device_unique_by_key_cleanup(cccl_device_unique_by_key_build_result_t* bld_ptr);

CCCL_C_EXTERN_C_END
