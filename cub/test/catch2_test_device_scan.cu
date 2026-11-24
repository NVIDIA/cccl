// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/config.cuh>

#include <cub/device/device_scan.cuh>
#include <cub/util_type.cuh>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cstddef>
#include <cstdint>

#include <vector_types.h>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include "catch2_test_numeric_helpers.cuh"
#include "catch2_test_scan_reference.cuh"
#include "cub_test_macros.h"
#include <c2h/buffer_generators.cuh>
#include <c2h/buffer_utils.cuh>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/custom_type.h>
#include <c2h/device_and_stream.h>
#include <c2h/extended_types.h>
#include <c2h/test_util_vec.h>
#include <c2h/utility.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanInit, device_inclusive_scan_with_init);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveSum, device_exclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveSum, device_inclusive_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2:3

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_pair<std::uint8_t, std::int32_t>, type_pair<std::int8_t>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_pair<std::int32_t>, type_pair<std::uint64_t>>;
#elif TEST_TYPES == 2
using full_type_list =
  c2h::type_list<type_pair<uchar3>,
                 type_pair<
#  if _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4_16a
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
                   >>;
#elif TEST_TYPES == 3
// clang-format off
using full_type_list = c2h::type_list<
type_pair<custom_t>
#if TEST_HALF_T()
, type_pair<half_t> // testing half
#endif // TEST_HALF_T()
#if TEST_BF_T()
, type_pair<bfloat16_t> // testing bf16
#endif // TEST_BF_T()
>;
// clang-format on
#endif

/**
 * @brief Input data generation mode
 */
enum class gen_data_t : int
{
  /// Uniform random data generation
  GEN_TYPE_RANDOM,
  /// Constant value as input data
  GEN_TYPE_CONST
};

CUB_TEST("Device scan works with all device interfaces", "[scan][device]", CUB_SMALL, full_type_list)
{
  using params   = params_t<TestType>;
  using input_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = std::int32_t;

  constexpr offset_t min_items = 1;
  constexpr offset_t max_items = 10'000'000;

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    1, // hits small copy path for bulk copies (below 16 bytes)
    10,
    1337,
    3000,
    1 * 31 * 128, // tile size for int64s for lookahead scan
    10'000, // a handful of tiles for lookahead scan
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  CAPTURE(num_items, c2h::type_name<input_t>(), c2h::type_name<output_t>());

  // Input data generation to test
  const gen_data_t data_gen_mode = GENERATE_COPY(gen_data_t::GEN_TYPE_RANDOM, gen_data_t::GEN_TYPE_CONST);

  const auto stream     = c2h::make_current_device_stream();
  const auto item_count = static_cast<std::size_t>(num_items);
  auto input_buffers    = [&] {
    if (data_gen_mode == gen_data_t::GEN_TYPE_RANDOM)
    {
      return c2h::gen_buffers<input_t>(stream, C2H_SEED(2), item_count);
    }

    input_t value{};
    init_default_constant(value);
    auto host_items   = c2h::make_host_buffer<input_t>(stream, item_count, value);
    auto device_items = c2h::make_device_buffer<input_t>(stream, host_items);
    return c2h::device_host_buffers<input_t>{cuda::std::move(device_items), cuda::std::move(host_items)};
  }();
  auto& in_items         = input_buffers.d_items;
  const auto& host_items = input_buffers.h_items;
  auto d_in_it           = in_items.data();

  // Skip DeviceScan::InclusiveSum and DeviceScan::ExclusiveSum tests for extended floating-point
  // types because of unbounded epsilon due to pseudo associativity of the addition operation over
  // floating point numbers
#if TEST_TYPES != 3
  SECTION("inclusive sum")
  {
    using op_t    = cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, input_t>;
    CAPTURE(c2h::type_name<op_t>(), c2h::type_name<accum_t>());

    // Prepare verification data
    auto expected_result = c2h::make_host_buffer<output_t>(stream, item_count, cuda::no_init);
    compute_inclusive_scan_reference(host_items.cbegin(), host_items.cend(), expected_result.begin(), op_t{}, accum_t{});

    // Run test
    auto out_result = c2h::make_device_buffer<output_t>(stream, item_count, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_out_it = out_result.data();
    device_inclusive_sum(d_in_it, d_out_it, num_items, stream);

    // Verify result
    REQUIRE_THAT_QUIET(out_result, c2h::Equals(stream, expected_result));

    // Run test in-place
    if constexpr (cuda::std::is_same_v<input_t, output_t>)
    {
      device_inclusive_sum(d_in_it, d_in_it, num_items, stream);

      // Verify result
      REQUIRE_THAT_QUIET(in_items, c2h::Equals(stream, expected_result));
    }
  }

  SECTION("exclusive sum")
  {
    using op_t    = cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, input_t>;
    CAPTURE(c2h::type_name<op_t>(), c2h::type_name<accum_t>());

    // Prepare verification data
    auto expected_result = c2h::make_host_buffer<output_t>(stream, item_count, cuda::no_init);
    compute_exclusive_scan_reference(host_items.cbegin(), host_items.cend(), expected_result.begin(), accum_t{}, op_t{});

    // Run test
    auto out_result = c2h::make_device_buffer<output_t>(stream, item_count, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_out_it = out_result.data();
    device_exclusive_sum(d_in_it, d_out_it, num_items, stream);

    // Verify result
    REQUIRE_THAT_QUIET(out_result, c2h::Equals(stream, expected_result));

    // Run test in-place
    if constexpr (cuda::std::is_same_v<input_t, output_t>)
    {
      device_exclusive_sum(d_in_it, d_in_it, num_items, stream);

      // Verify result
      REQUIRE_THAT_QUIET(in_items, c2h::Equals(stream, expected_result));
    }
  }
#endif

  SECTION("inclusive scan")
  {
    using op_t    = cuda::minimum<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, input_t>;
    CAPTURE(c2h::type_name<op_t>(), c2h::type_name<accum_t>());

    // Prepare verification data
    auto expected_result = c2h::make_host_buffer<output_t>(stream, item_count, cuda::no_init);
    compute_inclusive_scan_reference(
      host_items.cbegin(),
      host_items.cend(),
      expected_result.begin(),
      op_t{},
      cuda::std::numeric_limits<accum_t>::max());

    // Run test
    auto out_result = c2h::make_device_buffer<output_t>(stream, item_count, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_out_it = out_result.data();
    device_inclusive_scan(unwrap_it(d_in_it), unwrap_it(d_out_it), op_t{}, num_items, stream);

    // Verify result
    REQUIRE_THAT_QUIET(out_result, c2h::Equals(stream, expected_result));

    // Run test in-place
    if constexpr (cuda::std::is_same_v<input_t, output_t>)
    {
      device_inclusive_scan(unwrap_it(d_in_it), unwrap_it(d_in_it), op_t{}, num_items, stream);

      // Verify result
      REQUIRE_THAT_QUIET(in_items, c2h::Equals(stream, expected_result));
    }
  }

  SECTION("inclusive scan with init value")
  {
    using op_t    = cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, input_t>;
    CAPTURE(c2h::type_name<op_t>(), c2h::type_name<accum_t>());

    // Scan operator
    auto scan_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // Prepare verification data
    auto expected_result = c2h::make_host_buffer<output_t>(stream, item_count, cuda::no_init);

    // Run test
    auto out_result = c2h::make_device_buffer<output_t>(stream, item_count, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_out_it = out_result.data();
    accum_t init_value{};
    init_default_constant(init_value);
    compute_inclusive_scan_reference(
      host_items.cbegin(), host_items.cend(), expected_result.begin(), scan_op, init_value);

    device_inclusive_scan_with_init(unwrap_it(d_in_it), unwrap_it(d_out_it), scan_op, init_value, num_items, stream);

    // Verify result
    REQUIRE_THAT_QUIET(out_result, c2h::Equals(stream, expected_result));

    // Run test in-place
    if constexpr (cuda::std::is_same_v<input_t, output_t>)
    {
      device_inclusive_scan_with_init(unwrap_it(d_in_it), unwrap_it(d_in_it), scan_op, init_value, num_items, stream);

      // Verify result
      REQUIRE_THAT_QUIET(in_items, c2h::Equals(stream, expected_result));
    }
  }

  SECTION("exclusive scan")
  {
    using op_t    = cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, input_t>;
    CAPTURE(c2h::type_name<op_t>(), c2h::type_name<accum_t>());

    // Scan operator
    auto scan_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // Prepare verification data
    auto expected_result = c2h::make_host_buffer<output_t>(stream, item_count, cuda::no_init);
    compute_exclusive_scan_reference(
      host_items.cbegin(), host_items.cend(), expected_result.begin(), accum_t{}, scan_op);

    // Run test
    auto out_result = c2h::make_device_buffer<output_t>(stream, item_count, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_out_it      = out_result.data();
    using init_value_t = cub::detail::it_value_t<decltype(unwrap_it(d_out_it))>;
    device_exclusive_scan(unwrap_it(d_in_it), unwrap_it(d_out_it), scan_op, init_value_t{}, num_items, stream);

    // Verify result
    REQUIRE_THAT_QUIET(out_result, c2h::Equals(stream, expected_result));

    // Run test in-place
    if constexpr (cuda::std::is_same_v<input_t, output_t>)
    {
      device_exclusive_scan(unwrap_it(d_in_it), unwrap_it(d_in_it), scan_op, init_value_t{}, num_items, stream);

      // Verify result
      REQUIRE_THAT_QUIET(in_items, c2h::Equals(stream, expected_result));
    }
  }

  SECTION("exclusive scan with future-init value")
  {
    using op_t    = cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, input_t>;
    CAPTURE(c2h::type_name<op_t>(), c2h::type_name<accum_t>());

    // Scan operator
    auto scan_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // Prepare verification data
    accum_t init_value{};
    init_default_constant(init_value);
    auto expected_result = c2h::make_host_buffer<output_t>(stream, item_count, cuda::no_init);
    compute_exclusive_scan_reference(
      host_items.cbegin(), host_items.cend(), expected_result.begin(), init_value, scan_op);

    // Run test
    auto out_result = c2h::make_device_buffer<output_t>(stream, item_count, cuda::no_init);
    cuda::fill_bytes(stream, out_result, 0x5a);
    auto d_out_it      = out_result.data();
    using init_value_t = cub::detail::it_value_t<decltype(unwrap_it(d_out_it))>;
    auto d_initial_value =
      c2h::make_device_buffer<init_value_t>(stream, {static_cast<init_value_t>(*unwrap_it(&init_value))});
    const auto future_init_value = cub::FutureValue<init_value_t>(d_initial_value.data());
    device_exclusive_scan(unwrap_it(d_in_it), unwrap_it(d_out_it), scan_op, future_init_value, num_items, stream);

    // Verify result
    REQUIRE_THAT_QUIET(out_result, c2h::Equals(stream, expected_result));

    // Run test in-place
    if constexpr (cuda::std::is_same_v<input_t, output_t>)
    {
      device_exclusive_scan(unwrap_it(d_in_it), unwrap_it(d_in_it), scan_op, future_init_value, num_items, stream);

      // Verify result
      REQUIRE_THAT_QUIET(in_items, c2h::Equals(stream, expected_result));
    }
  }
}
