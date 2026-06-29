//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once
// NOLINTBEGIN(modernize-use-using)

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <cuda.h>
#include <stdint.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/transform.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

typedef struct cccl_device_binary_search_build_result_t
{
  cccl_device_transform_build_result_t transform;
  size_t op_state_size;
  size_t op_state_alignment;
} cccl_device_binary_search_build_result_t;

CCCL_C_API CUresult cccl_device_binary_search_build(
  cccl_device_binary_search_build_result_t* build,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

// Extended version with build configuration
CCCL_C_API CUresult cccl_device_binary_search_build_ex(
  cccl_device_binary_search_build_result_t* build,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_binary_search_compile(
  cccl_device_binary_search_build_result_t* build,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_binary_search_load(cccl_device_binary_search_build_result_t* build);

CCCL_C_API CUresult cccl_device_binary_search(
  cccl_device_binary_search_build_result_t build,
  cccl_iterator_t d_data,
  uint64_t num_items,
  cccl_iterator_t d_values,
  uint64_t num_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  CUstream stream);

CCCL_C_API CUresult cccl_device_binary_search_link_ltoir(
  cccl_device_binary_search_build_result_t* build,
  const void** input_blobs,
  const size_t* input_sizes,
  size_t num_inputs,
  const char* kernel_lowered_name,
  size_t values_value_size,
  size_t output_value_size,
  size_t op_state_size,
  size_t op_state_alignment,
  int cc_major,
  int cc_minor);

// Serializes a populated build_result into a self-describing byte buffer.
// On success *out_buf points to a heap allocation that the caller must free
// with cccl_aot_buffer_free, and *out_size holds its length. The build_result
// itself is not modified. CUlibrary/CUkernel handles are not serialized.
CCCL_C_API CUresult cccl_device_binary_search_serialize(
  const cccl_device_binary_search_build_result_t* build, void** out_buf, size_t* out_size);

// Reconstructs a build_result from a buffer produced by cccl_device_binary_search_serialize.
// On success build is populated as if by compile(); CUlibrary/CUkernel handles
// remain null until cccl_device_binary_search_load is called. On failure build is
// left unchanged and a non-success CUresult is returned.
CCCL_C_API CUresult
cccl_device_binary_search_deserialize(cccl_device_binary_search_build_result_t* build, const void* buf, size_t size);

CCCL_C_API CUresult cccl_device_binary_search_cleanup(cccl_device_binary_search_build_result_t* bld_ptr);

CCCL_C_EXTERN_C_END
// NOLINTEND(modernize-use-using)
