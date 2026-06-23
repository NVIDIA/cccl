// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_transform.cuh>

#include <algorithm>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceTransform::Transform, transform_many);

// Generic counts, deliberately including non-multiples of the 16-byte vectorized store width, to exercise the scalar
// tail of the ublkcp vectorized store path on aligned (c2h::device_vector) buffers.
#define GENERIC_COUNTS 0, 1, 15, 16, 17, 127, 129, 4095, 4097, 100'000

template <typename Out>
struct cast_to
{
  template <typename T>
  __host__ __device__ Out operator()(T v) const
  {
    return static_cast<Out>(v);
  }
};

// Narrowing widths (e.g. uint32 -> uint8) drive the multi-int4-load gather; widening (uint8 -> uint32) drives the
// sub-16-byte load. Same-width is already covered by catch2_test_device_transform.cu's BabelStream add.
C2H_TEST("DeviceTransform::Transform vectorized store narrowing to uint8",
         "[device][transform]",
         c2h::type_list<std::uint16_t, std::uint32_t, std::uint64_t>)
{
  using in_t               = c2h::get<0, TestType>;
  using out_t              = std::uint8_t;
  using offset_t           = cuda::std::int64_t;
  const offset_t num_items = GENERATE(GENERIC_COUNTS);
  CAPTURE(c2h::type_name<in_t>(), num_items);

  c2h::device_vector<in_t> in(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), in);

  c2h::device_vector<out_t> result(num_items, thrust::no_init);
  transform_many(cuda::std::make_tuple(in.begin()), result.begin(), num_items, cast_to<out_t>{});

  c2h::host_vector<in_t> in_h = in;
  c2h::host_vector<out_t> reference_h(num_items, thrust::no_init);
  std::transform(in_h.begin(), in_h.end(), reference_h.begin(), cast_to<out_t>{});
  REQUIRE(reference_h == result);
}

C2H_TEST("DeviceTransform::Transform vectorized store widening from uint8",
         "[device][transform]",
         c2h::type_list<std::uint16_t, std::uint32_t, std::uint64_t>)
{
  using in_t               = std::uint8_t;
  using out_t              = c2h::get<0, TestType>;
  using offset_t           = cuda::std::int64_t;
  const offset_t num_items = GENERATE(GENERIC_COUNTS);
  CAPTURE(c2h::type_name<out_t>(), num_items);

  c2h::device_vector<in_t> in(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), in);

  c2h::device_vector<out_t> result(num_items, thrust::no_init);
  transform_many(cuda::std::make_tuple(in.begin()), result.begin(), num_items, cast_to<out_t>{});

  c2h::host_vector<in_t> in_h = in;
  c2h::host_vector<out_t> reference_h(num_items, thrust::no_init);
  std::transform(in_h.begin(), in_h.end(), reference_h.begin(), cast_to<out_t>{});
  REQUIRE(reference_h == result);
}

struct ublkcp_store_vec_size_2_selector
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> cub::TransformPolicy
  {
    auto async              = cub::TransformAsyncCopyPolicy{};
    async.threads_per_block = 256;
    async.store_vec_size    = 2;
    const auto algorithm =
      (cc < ::cuda::compute_capability{9, 0}) ? cub::TransformAlgorithm::prefetch : cub::TransformAlgorithm::ublkcp;
    return {64 * 1024, algorithm, cub::TransformPrefetchPolicy{256}, {}, async};
  }
};

C2H_TEST("DeviceTransform::Transform tunable narrower store_vec_size", "[device][transform]")
{
  using in_t                         = std::uint32_t;
  using out_t                        = std::uint8_t;
  const cuda::std::int64_t num_items = GENERATE(GENERIC_COUNTS);
  CAPTURE(num_items);

  c2h::device_vector<in_t> in(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), in);

  c2h::device_vector<out_t> result(num_items, thrust::no_init);
  auto env = cuda::execution::tune(ublkcp_store_vec_size_2_selector{});
  REQUIRE(cudaSuccess
          == cub::DeviceTransform::Transform(
            cuda::std::make_tuple(in.begin()), result.begin(), num_items, cast_to<out_t>{}, env));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<in_t> in_h = in;
  c2h::host_vector<out_t> reference_h(num_items, thrust::no_init);
  std::transform(in_h.begin(), in_h.end(), reference_h.begin(), cast_to<out_t>{});
  REQUIRE(reference_h == result);
}
