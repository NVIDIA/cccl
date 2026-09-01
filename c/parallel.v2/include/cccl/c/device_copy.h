//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#pragma once
// NOLINTBEGIN(modernize-use-using)

#include <cuda.h>

#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

typedef enum cccl_device_copy_axis_metadata_kind_t
{
  // The value for this axis is supplied by the runtime view passed to cccl_device_copy.
  // The build-time metadata value must be 0 and is ignored.
  CCCL_DEVICE_COPY_AXIS_RUNTIME = 0,
  // The value for this axis is a compile-time constant in the generated mdspan type
  // or mapping. The build-time metadata value is the constant axis value. Reserved
  // for a later implementation; the current builder accepts runtime metadata only.
  CCCL_DEVICE_COPY_AXIS_STATIC = 1,
} cccl_device_copy_axis_metadata_kind_t;

typedef struct cccl_device_copy_axis_metadata_t
{
  cccl_device_copy_axis_metadata_kind_t kind;
  int64_t value;
} cccl_device_copy_axis_metadata_t;

typedef enum cccl_device_copy_layout_kind_t
{
  // cuda::std::layout_right. Runtime stride pointers are ignored.
  CCCL_DEVICE_COPY_LAYOUT_RIGHT = 0,
  // cuda::std::layout_left. Runtime stride pointers are ignored.
  CCCL_DEVICE_COPY_LAYOUT_LEFT = 1,
  // cuda::std::layout_stride. Strides are in elements and must be non-negative.
  // Reserved for a later implementation.
  CCCL_DEVICE_COPY_LAYOUT_STRIDE = 2,
  // cuda::layout_stride_relaxed. Strides are in elements and may be negative.
  CCCL_DEVICE_COPY_LAYOUT_STRIDE_RELAXED = 3,
} cccl_device_copy_layout_kind_t;

typedef struct cccl_device_copy_view_build_t
{
  cccl_device_copy_layout_kind_t layout;
  // Array of rank stride metadata entries for strided layouts. Ignored for
  // layout_left/layout_right and may be NULL in that case. Current implementation
  // requires all layout_stride_relaxed stride metadata entries to be runtime.
  const cccl_device_copy_axis_metadata_t* strides;
} cccl_device_copy_view_build_t;

typedef struct cccl_device_copy_build_spec_t
{
  cccl_type_info value_type;
  // Current implementation accepts positive ranks.
  size_t rank;
  // Array of rank extent metadata entries. Static extent values must be
  // non-negative. Runtime extent values are supplied by runtime views. Current
  // implementation requires all entries to be runtime metadata.
  const cccl_device_copy_axis_metadata_t* shape;
  cccl_device_copy_view_build_t source;
  cccl_device_copy_view_build_t destination;
} cccl_device_copy_build_spec_t;

typedef struct cccl_device_copy_source_view_t
{
  const void* data;
  // DLPack-style byte offset from data to the first logical element. The
  // effective address data + byte_offset must satisfy value_type.alignment.
  uint64_t byte_offset;
  // Host pointer to rank extents. Required when any shape axis is runtime.
  // Valid for the duration of the API call; data lifetime is governed by stream
  // ordering and must extend until the copy work has completed.
  const int64_t* shape;
  // Host pointer to rank strides in elements. Required for runtime stride axes
  // of strided layouts; ignored for layout_left/layout_right.
  const int64_t* strides;
} cccl_device_copy_source_view_t;

typedef struct cccl_device_copy_destination_view_t
{
  void* data;
  // DLPack-style byte offset from data to the first logical element. The
  // effective address data + byte_offset must satisfy value_type.alignment.
  uint64_t byte_offset;
  // Host pointer to rank extents. Required when any shape axis is runtime.
  // Valid for the duration of the API call; data lifetime is governed by stream
  // ordering and must extend until the copy work has completed.
  const int64_t* shape;
  // Host pointer to rank strides in elements. Required for runtime stride axes
  // of strided layouts; ignored for layout_left/layout_right.
  const int64_t* strides;
} cccl_device_copy_destination_view_t;

typedef struct cccl_device_copy_build_result_t
{
  int cc;
  void* payload;
  size_t payload_size;
  void* jit_compiler;
  void* copy_fn;
  cccl_type_info value_type;
  size_t rank;
  cccl_device_copy_layout_kind_t source_layout;
  cccl_device_copy_layout_kind_t destination_layout;
} cccl_device_copy_build_result_t;

CCCL_C_API CUresult cccl_device_copy_build_ex(
  cccl_device_copy_build_result_t* build_ptr,
  cccl_device_copy_build_spec_t spec,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* build_config);

CCCL_C_API CUresult cccl_device_copy_build(
  cccl_device_copy_build_result_t* build_ptr,
  cccl_device_copy_build_spec_t spec,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

CCCL_C_API CUresult cccl_device_copy(
  cccl_device_copy_build_result_t build,
  cccl_device_copy_source_view_t source,
  cccl_device_copy_destination_view_t destination,
  CUstream stream);

CCCL_C_API CUresult cccl_device_copy_cleanup(cccl_device_copy_build_result_t* build_ptr);

CCCL_C_EXTERN_C_END
// NOLINTEND(modernize-use-using)
