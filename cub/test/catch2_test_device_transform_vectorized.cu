// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_transform.cuh>

#include <thrust/sequence.h>

#include <cuda/iterator>
#include <cuda/std/functional>

#include <algorithm>
#include <numeric>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

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
CUB_TEST("DeviceTransform::Transform vectorized store narrowing to uint8",
         "[device][transform]",
         CUB_SMALL,
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

CUB_TEST("DeviceTransform::Transform vectorized store widening from uint8",
         "[device][transform]",
         CUB_SMALL,
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

struct plus_one_t
{
  __host__ __device__ int operator()(int value) const
  {
    return value + 1;
  }
};

// The input is a transform iterator over a pointer, so the kernel loads the stored elements and the functor runs on
// that copy. Counts include a partial last tile and a pointer that is not 16-byte aligned.
CUB_TEST("DeviceTransform vectorized load of a contiguous transform iterator", "[device][transform]", CUB_SMALL)
{
  const int num_items = GENERATE(4096, 4097, 100000);
  c2h::device_vector<int> in(num_items + 1);
  thrust::sequence(c2h::device_policy, in.begin(), in.end());
  auto* d_in = thrust::raw_pointer_cast(in.data());

  c2h::device_vector<int> result(num_items, thrust::no_init);
  transform_many(cuda::std::make_tuple(cuda::transform_iterator(d_in, plus_one_t{})),
                 result.begin(),
                 num_items,
                 cuda::std::identity{});

  c2h::host_vector<int> reference(num_items);
  std::iota(reference.begin(), reference.end(), 1);
  REQUIRE(reference == result);

  c2h::device_vector<int> shifted(num_items - 1, thrust::no_init);
  transform_many(cuda::std::make_tuple(cuda::transform_iterator(d_in + 1, plus_one_t{})),
                 shifted.begin(),
                 num_items - 1,
                 cuda::std::identity{});

  c2h::host_vector<int> shifted_reference(num_items - 1);
  std::iota(shifted_reference.begin(), shifted_reference.end(), 2);
  REQUIRE(shifted_reference == shifted);
}

// Widening changes the stored type and the value type. The kernel has to load the stored bytes, then apply the functor.
CUB_TEST("DeviceTransform contiguous transform iterator widens the stored type", "[device][transform]", CUB_SMALL)
{
  const int num_items = GENERATE(4096, 4097, 100000);
  c2h::device_vector<std::uint8_t> in(num_items + 1);
  thrust::sequence(c2h::device_policy, in.begin(), in.end());
  auto* d_in = thrust::raw_pointer_cast(in.data());

  c2h::device_vector<int> result(num_items, thrust::no_init);
  transform_many(cuda::std::make_tuple(cuda::transform_iterator(d_in, cast_to<int>{})),
                 result.begin(),
                 num_items,
                 cuda::std::identity{});

  c2h::host_vector<std::uint8_t> in_h = in;
  c2h::host_vector<int> reference(num_items);
  std::transform(in_h.begin(), in_h.begin() + num_items, reference.begin(), cast_to<int>{});
  REQUIRE(reference == result);

  c2h::device_vector<int> shifted(num_items - 1, thrust::no_init);
  transform_many(cuda::std::make_tuple(cuda::transform_iterator(d_in + 1, cast_to<int>{})),
                 shifted.begin(),
                 num_items - 1,
                 cuda::std::identity{});

  c2h::host_vector<int> shifted_reference(num_items - 1);
  std::transform(in_h.begin() + 1, in_h.begin() + num_items, shifted_reference.begin(), cast_to<int>{});
  REQUIRE(shifted_reference == shifted);
}

#if TEST_LAUNCH == 0
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

CUB_TEST("DeviceTransform::Transform tunable narrower store_vec_size", "[device][transform]", CUB_SMALL)
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
#endif // TEST_LAUNCH == 0
