//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once
// NOLINTBEGIN(modernize-use-using)

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <cuda.h>
#include <stdbool.h>
#include <stdint.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

//! Reduction over runs of equal adjacent keys (HostJIT backend).
//!
//! Mirrors `cub::DeviceReduce::ReduceByKey`: no equality operator and no initial value, key equality
//! comes from the key type, and a run's representative key is its last key.
typedef struct cccl_device_reduce_by_key_build_result_t
{
  int cc;
  void* payload;
  size_t payload_size;
  void* jit_compiler;
  void* reduce_by_key_fn;
} cccl_device_reduce_by_key_build_result_t;

CCCL_C_API CUresult cccl_device_reduce_by_key_build(
  cccl_device_reduce_by_key_build_result_t* build,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

// Extended version with build configuration
CCCL_C_API CUresult cccl_device_reduce_by_key_build_ex(
  cccl_device_reduce_by_key_build_result_t* build,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

//! Runs the reduction; the outputs are sized for the worst case by the caller and the run count stays
//! on the device.
CCCL_C_API CUresult cccl_device_reduce_by_key(
  cccl_device_reduce_by_key_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_unique_out,
  cccl_iterator_t d_aggregates_out,
  cccl_iterator_t d_num_runs_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream);

CCCL_C_API CUresult cccl_device_reduce_by_key_cleanup(cccl_device_reduce_by_key_build_result_t* build_ptr);

CCCL_C_EXTERN_C_END

// NOLINTEND(modernize-use-using)
