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

//! Device-wide scan over runs of equal adjacent keys.
//!
//! Mirrors `cub::DeviceScan::InclusiveScanByKey` / `ExclusiveScanByKey`. The `init` value is only
//! consumed by the exclusive form; the inclusive form has no init parameter in CUB either.
typedef struct cccl_device_scan_by_key_build_result_t
{
  int cc;
  void* payload;
  size_t payload_size;
  cccl_payload_kind_t payload_kind;
  CUlibrary library;
  cccl_type_info key_type;
  cccl_type_info input_type; // values input type (name follows scan.h)
  cccl_type_info input_value_type;
  cccl_type_info output_value_type;
  cccl_type_info output_type; // values output type (name follows scan.h)
  cccl_type_info accumulator_type;
  CUkernel init_kernel;
  CUkernel scan_kernel;
  bool force_inclusive;
  cccl_init_kind_t init_kind;
  size_t description_bytes_per_tile;
  size_t payload_bytes_per_tile;
  void* runtime_policy;
  size_t runtime_policy_size;
  char* init_kernel_lowered_name;
  char* scan_kernel_lowered_name;
} cccl_device_scan_by_key_build_result_t;

CCCL_C_API CUresult cccl_device_scan_by_key_build(
  cccl_device_scan_by_key_build_result_t* build_ptr,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t op,
  cccl_op_t equality_op,
  cccl_type_info init,
  bool force_inclusive,
  cccl_init_kind_t init_kind,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

// Extended version with build configuration
CCCL_C_API CUresult cccl_device_scan_by_key_build_ex(
  cccl_device_scan_by_key_build_result_t* build_ptr,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t op,
  cccl_op_t equality_op,
  cccl_type_info init,
  bool force_inclusive,
  cccl_init_kind_t init_kind,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_scan_by_key_compile(
  cccl_device_scan_by_key_build_result_t* build_ptr,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t op,
  cccl_op_t equality_op,
  cccl_type_info init,
  bool force_inclusive,
  cccl_init_kind_t init_kind,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_scan_by_key_load(cccl_device_scan_by_key_build_result_t* build_ptr);

//! Inclusive scan by key. CUB's inclusive by-key scan takes no init value, so neither does this.
CCCL_C_API CUresult cccl_device_inclusive_scan_by_key(
  cccl_device_scan_by_key_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_op_t equality_op,
  CUstream stream);

//! Exclusive scan by key. The init value is applied at the head of *every* key run.
CCCL_C_API CUresult cccl_device_exclusive_scan_by_key(
  cccl_device_scan_by_key_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_op_t equality_op,
  cccl_value_t init,
  CUstream stream);

CCCL_C_API CUresult cccl_device_scan_by_key_link_ltoir(
  cccl_device_scan_by_key_build_result_t* build,
  const void** input_blobs,
  const size_t* input_sizes,
  size_t num_inputs);

CCCL_C_API CUresult
cccl_device_scan_by_key_serialize(const cccl_device_scan_by_key_build_result_t* build, void** out_buf, size_t* out_size);

// Reconstructs a build_result from a buffer produced by cccl_device_scan_by_key_serialize.
// Kernel handles remain null until cccl_device_scan_by_key_load is called.
CCCL_C_API CUresult
cccl_device_scan_by_key_deserialize(cccl_device_scan_by_key_build_result_t* build, const void* buf, size_t size);

CCCL_C_API CUresult cccl_device_scan_by_key_cleanup(cccl_device_scan_by_key_build_result_t* bld_ptr);

CCCL_C_EXTERN_C_END

// NOLINTEND(modernize-use-using)
