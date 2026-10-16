//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include <catch2/catch_test_macros.hpp>
#include <cccl/c/device_copy.h>

namespace
{
extern "C" int validation_copy_fn(
  const void*,
  unsigned long long,
  const int64_t*,
  const int64_t*,
  void*,
  unsigned long long,
  const int64_t*,
  const int64_t*,
  void*)
{
  return 0;
}

cccl_device_copy_build_result_t make_validation_build(
  cccl_device_copy_axis_metadata_t* shape,
  cccl_device_copy_axis_metadata_t* source_strides,
  cccl_device_copy_axis_metadata_t* destination_strides,
  cccl_device_copy_layout_kind_t source_layout      = CCCL_DEVICE_COPY_LAYOUT_STRIDE,
  cccl_device_copy_layout_kind_t destination_layout = CCCL_DEVICE_COPY_LAYOUT_STRIDE)
{
  cccl_device_copy_build_result_t build{};
  build.copy_fn             = reinterpret_cast<void*>(&validation_copy_fn);
  build.value_type          = cccl_type_info{sizeof(std::int32_t), alignof(std::int32_t), CCCL_STORAGE};
  build.rank                = 1;
  build.shape               = shape;
  build.source_strides      = source_strides;
  build.destination_strides = destination_strides;
  build.source_layout       = source_layout;
  build.destination_layout  = destination_layout;
  return build;
}

cccl_device_copy_source_view_t make_source_view(const int64_t* shape, const int64_t* strides)
{
  return cccl_device_copy_source_view_t{reinterpret_cast<const void*>(0x1000), 0, shape, strides};
}

cccl_device_copy_destination_view_t make_destination_view(const int64_t* shape, const int64_t* strides)
{
  return cccl_device_copy_destination_view_t{reinterpret_cast<void*>(0x2000), 0, shape, strides};
}
} // namespace

CATCH_TEST_CASE("DeviceCopy C API accepts valid runtime metadata", "[device_copy]")
{
  cccl_device_copy_axis_metadata_t runtime_metadata[] = {{CCCL_DEVICE_COPY_AXIS_RUNTIME, 0}};
  auto build = make_validation_build(runtime_metadata, runtime_metadata, runtime_metadata);

  const int64_t shape[]   = {4};
  const int64_t strides[] = {1};

  CATCH_REQUIRE(
    cccl_device_copy(build, make_source_view(shape, strides), make_destination_view(shape, strides), nullptr)
    == CUDA_SUCCESS);
}

CATCH_TEST_CASE("DeviceCopy C API accepts runtime extent that matches static build metadata", "[device_copy]")
{
  cccl_device_copy_axis_metadata_t static_shape[]    = {{CCCL_DEVICE_COPY_AXIS_STATIC, 4}};
  cccl_device_copy_axis_metadata_t runtime_strides[] = {{CCCL_DEVICE_COPY_AXIS_RUNTIME, 0}};
  auto build = make_validation_build(static_shape, runtime_strides, runtime_strides);

  const int64_t shape[]   = {4};
  const int64_t strides[] = {1};

  CATCH_REQUIRE(
    cccl_device_copy(build, make_source_view(shape, strides), make_destination_view(shape, strides), nullptr)
    == CUDA_SUCCESS);
}

CATCH_TEST_CASE("DeviceCopy C API rejects runtime shape mismatch", "[device_copy]")
{
  cccl_device_copy_axis_metadata_t runtime_metadata[] = {{CCCL_DEVICE_COPY_AXIS_RUNTIME, 0}};
  auto build = make_validation_build(runtime_metadata, runtime_metadata, runtime_metadata);

  const int64_t source_shape[]      = {4};
  const int64_t destination_shape[] = {5};
  const int64_t strides[]           = {1};

  CATCH_REQUIRE(
    cccl_device_copy(
      build, make_source_view(source_shape, strides), make_destination_view(destination_shape, strides), nullptr)
    == CUDA_ERROR_INVALID_VALUE);
}

CATCH_TEST_CASE("DeviceCopy C API rejects runtime extent that differs from static build metadata", "[device_copy]")
{
  cccl_device_copy_axis_metadata_t static_shape[]    = {{CCCL_DEVICE_COPY_AXIS_STATIC, 4}};
  cccl_device_copy_axis_metadata_t runtime_strides[] = {{CCCL_DEVICE_COPY_AXIS_RUNTIME, 0}};
  auto build = make_validation_build(static_shape, runtime_strides, runtime_strides);

  const int64_t shape[]   = {5};
  const int64_t strides[] = {1};

  CATCH_REQUIRE(
    cccl_device_copy(build, make_source_view(shape, strides), make_destination_view(shape, strides), nullptr)
    == CUDA_ERROR_INVALID_VALUE);
}

CATCH_TEST_CASE("DeviceCopy C API rejects negative strides for layout_stride", "[device_copy]")
{
  cccl_device_copy_axis_metadata_t runtime_metadata[] = {{CCCL_DEVICE_COPY_AXIS_RUNTIME, 0}};
  auto build = make_validation_build(runtime_metadata, runtime_metadata, runtime_metadata);

  const int64_t shape[]          = {4};
  const int64_t source_strides[] = {-1};
  const int64_t output_strides[] = {1};

  CATCH_REQUIRE(cccl_device_copy(
                  build, make_source_view(shape, source_strides), make_destination_view(shape, output_strides), nullptr)
                == CUDA_ERROR_INVALID_VALUE);
}
