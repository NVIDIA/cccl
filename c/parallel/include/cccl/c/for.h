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
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

typedef struct cccl_device_for_build_result_t
{
  int cc;
  void* payload;
  size_t payload_size;
  cccl_payload_kind_t payload_kind;
  CUlibrary library;
  CUkernel static_kernel;
  char* static_kernel_lowered_name;
} cccl_device_for_build_result_t;

CCCL_C_API CUresult cccl_device_for_build(
  cccl_device_for_build_result_t* build,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

// Extended version with build configuration
CCCL_C_API CUresult cccl_device_for_build_ex(
  cccl_device_for_build_result_t* build,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_for_compile(
  cccl_device_for_build_result_t* build,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_for_load(cccl_device_for_build_result_t* build);

CCCL_C_API CUresult cccl_device_for_link_ltoir(
  cccl_device_for_build_result_t* build, const void** input_blobs, const size_t* input_sizes, size_t num_inputs);

CCCL_C_API CUresult cccl_device_for(
  cccl_device_for_build_result_t build, cccl_iterator_t d_data, uint64_t num_items, cccl_op_t op, CUstream stream);

// Serializes a populated build_result into a self-describing byte buffer.
// On success *out_buf points to a heap allocation that the caller must free
// with cccl_aot_buffer_free, and *out_size holds its length. The build_result
// itself is not modified. CUlibrary/CUkernel handles are not serialized.
CCCL_C_API CUresult
cccl_device_for_serialize(const cccl_device_for_build_result_t* build, void** out_buf, size_t* out_size);

// Reconstructs a build_result from a buffer produced by cccl_device_for_serialize.
// On success build is populated as if by compile(); CUlibrary/CUkernel handles
// remain null until cccl_device_for_load is called. If the deserialized payload
// kind is CCCL_PAYLOAD_LTOIR, cccl_device_for_link_ltoir must be called before
// cccl_device_for_load. On failure build is left unchanged and a non-success
// CUresult is returned.
CCCL_C_API CUresult cccl_device_for_deserialize(cccl_device_for_build_result_t* build, const void* buf, size_t size);

CCCL_C_API CUresult cccl_device_for_cleanup(cccl_device_for_build_result_t* bld_ptr);

CCCL_C_EXTERN_C_END
// NOLINTEND(modernize-use-using)
