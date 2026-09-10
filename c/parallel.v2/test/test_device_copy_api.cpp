//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>
#include <cccl/c/device_copy.h>

namespace
{
using build_ex_fn_t = CUresult (*)(
  cccl_device_copy_build_result_t*,
  cccl_device_copy_build_spec_t,
  int,
  int,
  const char*,
  const char*,
  const char*,
  const char*,
  cccl_build_config*);

using build_fn_t = CUresult (*)(
  cccl_device_copy_build_result_t*,
  cccl_device_copy_build_spec_t,
  int,
  int,
  const char*,
  const char*,
  const char*,
  const char*);

using copy_fn_t = CUresult (*)(
  cccl_device_copy_build_result_t, cccl_device_copy_source_view_t, cccl_device_copy_destination_view_t, CUstream);

using cleanup_fn_t = CUresult (*)(cccl_device_copy_build_result_t*);

static_assert(std::is_same_v<decltype(&cccl_device_copy_build_ex), build_ex_fn_t>);
static_assert(std::is_same_v<decltype(&cccl_device_copy_build), build_fn_t>);
static_assert(std::is_same_v<decltype(&cccl_device_copy), copy_fn_t>);
static_assert(std::is_same_v<decltype(&cccl_device_copy_cleanup), cleanup_fn_t>);

static_assert(std::is_standard_layout_v<cccl_device_copy_axis_metadata_t>);
static_assert(std::is_standard_layout_v<cccl_device_copy_view_build_t>);
static_assert(std::is_standard_layout_v<cccl_device_copy_build_spec_t>);
static_assert(std::is_standard_layout_v<cccl_device_copy_source_view_t>);
static_assert(std::is_standard_layout_v<cccl_device_copy_destination_view_t>);
static_assert(std::is_standard_layout_v<cccl_device_copy_build_result_t>);

static_assert(std::is_trivially_copyable_v<cccl_device_copy_axis_metadata_t>);
static_assert(std::is_trivially_copyable_v<cccl_device_copy_view_build_t>);
static_assert(std::is_trivially_copyable_v<cccl_device_copy_build_spec_t>);
static_assert(std::is_trivially_copyable_v<cccl_device_copy_source_view_t>);
static_assert(std::is_trivially_copyable_v<cccl_device_copy_destination_view_t>);
static_assert(std::is_trivially_copyable_v<cccl_device_copy_build_result_t>);
} // namespace

CATCH_TEST_CASE("DeviceCopy C API describes build metadata and runtime views", "[device_copy][api]")
{
  const cccl_device_copy_axis_metadata_t shape[] = {
    {CCCL_DEVICE_COPY_AXIS_STATIC, 4},
    {CCCL_DEVICE_COPY_AXIS_RUNTIME, 0},
  };
  const cccl_device_copy_axis_metadata_t source_strides[] = {
    {CCCL_DEVICE_COPY_AXIS_STATIC, 1},
    {CCCL_DEVICE_COPY_AXIS_RUNTIME, 0},
  };
  const cccl_device_copy_axis_metadata_t destination_strides[] = {
    {CCCL_DEVICE_COPY_AXIS_RUNTIME, 0},
    {CCCL_DEVICE_COPY_AXIS_STATIC, 4},
  };

  const cccl_type_info value_type{sizeof(std::int32_t), alignof(std::int32_t), CCCL_INT32};
  const cccl_device_copy_build_spec_t spec{
    value_type,
    2,
    shape,
    {CCCL_DEVICE_COPY_LAYOUT_STRIDE_RELAXED, source_strides},
    {CCCL_DEVICE_COPY_LAYOUT_STRIDE, destination_strides}};

  CATCH_REQUIRE(spec.rank == 2);
  CATCH_REQUIRE(spec.value_type.type == CCCL_INT32);
  CATCH_REQUIRE(spec.shape[0].kind == CCCL_DEVICE_COPY_AXIS_STATIC);
  CATCH_REQUIRE(spec.shape[1].kind == CCCL_DEVICE_COPY_AXIS_RUNTIME);
  CATCH_REQUIRE(spec.source.layout == CCCL_DEVICE_COPY_LAYOUT_STRIDE_RELAXED);
  CATCH_REQUIRE(spec.destination.layout == CCCL_DEVICE_COPY_LAYOUT_STRIDE);
  CATCH_REQUIRE(spec.source.strides[1].kind == CCCL_DEVICE_COPY_AXIS_RUNTIME);
  CATCH_REQUIRE(spec.destination.strides[1].value == 4);

  const std::int64_t runtime_shape[]   = {4, 8};
  const std::int64_t runtime_strides[] = {1, 4};
  const cccl_device_copy_source_view_t source{nullptr, 0, runtime_shape, runtime_strides};
  const cccl_device_copy_destination_view_t destination{nullptr, 0, runtime_shape, runtime_strides};

  CATCH_REQUIRE(source.shape == runtime_shape);
  CATCH_REQUIRE(source.strides == runtime_strides);
  CATCH_REQUIRE(destination.shape == runtime_shape);
  CATCH_REQUIRE(destination.strides == runtime_strides);
}
