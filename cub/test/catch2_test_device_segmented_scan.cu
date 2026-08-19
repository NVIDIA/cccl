// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <cuda/__algorithm/copy.h>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/devices>
#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include <cub_test_macros_lightweight.h>
#include <cuda_runtime_api.h>

#include <c2h/buffer_generators.cuh>
#include <c2h/checked_memory_resource.cuh>
#include <c2h/custom_type.h>
#include <c2h/extended_types.h>
#include <c2h/test_util_vec.h>
#include <c2h/utility.h>
#include <catch2_test_device_segmented_scan_utils.cuh>
#include <catch2_test_launch_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedSum, device_inclusive_segmented_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::ExclusiveSegmentedSum, device_exclusive_segmented_sum);

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScan, device_inclusive_segmented_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::ExclusiveSegmentedScan, device_exclusive_segmented_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScanInit, device_inclusive_segmented_scan_with_init);

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2:3

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

struct ExtendedFloatSum
{
  template <class T>
  __host__ __device__ T operator()(T a, T b) const
  {
    T result{};
    result.__x = a.raw() + b.raw();
    return result;
  }

#if TEST_HALF_T()
  __host__ __device__ __half operator()(__half a, __half b) const
  {
    uint16_t result = this->operator()(half_t{a}, half_t(b)).raw();
    return reinterpret_cast<__half&>(result);
  }
#endif // TEST_HALF_T()

#if TEST_BF_T()
  __device__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const
  {
    uint16_t result = this->operator()(bfloat16_t{a}, bfloat16_t(b)).raw();
    return reinterpret_cast<__nv_bfloat16&>(result);
  }
#endif // TEST_BF_T()
};

template <class It>
inline It unwrap_it(It it)
{
  return it;
}

#if TEST_HALF_T()
inline __half* unwrap_it(half_t* it)
{
  return reinterpret_cast<__half*>(it);
}

template <class OffsetT>
inline cuda::constant_iterator<__half, OffsetT> unwrap_it(cuda::constant_iterator<half_t, OffsetT> it)
{
  half_t wrapped_val = *it;
  __half val         = wrapped_val.operator __half();
  return cuda::constant_iterator<__half, OffsetT>(val);
}
#endif // TEST_HALF_T()

#if TEST_BF_T()
inline __nv_bfloat16* unwrap_it(bfloat16_t* it)
{
  return reinterpret_cast<__nv_bfloat16*>(it);
}

template <class OffsetT>
cuda::constant_iterator<__nv_bfloat16, OffsetT> inline unwrap_it(cuda::constant_iterator<bfloat16_t, OffsetT> it)
{
  bfloat16_t wrapped_val = *it;
  __nv_bfloat16 val      = wrapped_val.operator __nv_bfloat16();
  return cuda::constant_iterator<__nv_bfloat16, OffsetT>(val);
}
#endif // TEST_BF_T()

template <class WrappedItT, class ItT = decltype(unwrap_it(std::declval<WrappedItT>()))>
std::integral_constant<bool, !std::is_same_v<WrappedItT, ItT>> inline reference_extended_fp(WrappedItT)
{
  return {};
}

inline constexpr ExtendedFloatSum unwrap_op(std::true_type, ::cuda::std::plus<>)
{
  return {};
}

template <bool V, class OpT>
inline constexpr OpT unwrap_op(std::integral_constant<bool, V>, OpT op)
{
  return op;
}

template <typename T>
inline void init_default_constant(T& val, int element_val = 2)
{
  val = T{static_cast<T>(element_val)};
}

template <template <typename> class... Policies>
inline void init_default_constant(c2h::custom_type_t<Policies...>& val, int element_val = 2)
{
  val.key = static_cast<size_t>(element_val);
  val.val = static_cast<size_t>(element_val);
}

inline void init_default_constant(uchar3& val, int element_val = 2)
{
  const auto element_init = static_cast<unsigned char>(element_val);
  val                     = uchar3{element_init, element_init, element_init};
}

_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
inline void init_default_constant(ulonglong4& val, int element_val = 2)
{
  const auto element_init = static_cast<unsigned long long>(element_val);
  val                     = ulonglong4{element_init, element_init, element_init, element_init};
}
_CCCL_SUPPRESS_DEPRECATED_POP

#if _CCCL_CTK_AT_LEAST(13, 0)
inline void init_default_constant(ulonglong4_16a& val, int element_val = 2)
{
  const auto element_init = static_cast<unsigned long long>(element_val);
  val                     = ulonglong4_16a{element_init, element_init, element_init, element_init};
}
#endif // _CCCL_CTK_AT_LEAST(13, 0)

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_pair<std::uint8_t>, type_pair<std::int8_t, std::int32_t>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_pair<std::int32_t>, type_pair<std::int64_t>>;
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

using segmented_scan_test::copy_to_host;
using segmented_scan_test::current_device;
using segmented_scan_test::make_host_buffer;

template <typename ValueT, typename OffsetT>
bool check_segment(const ValueT* h_output, const ValueT* h_ref, OffsetT begin_offset, OffsetT end_offset)
{
  using value_t = ValueT;

  bool correct = true;
  for (OffsetT pos = begin_offset; pos < end_offset; ++pos)
  {
    if constexpr (cuda::std::is_floating_point_v<value_t>)
    {
      value_t ref_v  = h_ref[pos];
      value_t act_v  = h_output[pos];
      value_t diff   = (ref_v - act_v);
      value_t adiff  = (diff > value_t{0}) ? diff : -diff;
      value_t ref_av = (ref_v > value_t{0}) ? ref_v : -ref_v;
      value_t act_av = (act_v > value_t{0}) ? act_v : -act_v;

      value_t eps = ::cuda::std::numeric_limits<value_t>::epsilon();
      correct     = correct && (adiff < 3 * eps + 2 * eps * (::cuda::std::max(ref_av, act_av)));
    }
    else if constexpr (cuda::std::is_same_v<value_t, half_t> || cuda::std::is_same_v<value_t, bfloat16_t>)
    {
      float ref_v = h_ref[pos];
      float act_v = h_output[pos];
      if (cuda::std::isfinite(ref_v) && cuda::std::isfinite(act_v))
      {
        float diff   = (ref_v - act_v);
        float adiff  = (diff > float{0}) ? diff : -diff;
        float ref_av = (ref_v > float{0}) ? ref_v : -ref_v;
        float act_av = (act_v > float{0}) ? act_v : -act_v;

        float eps = float{1} / float{128};
        correct   = correct && (adiff < 3 * eps + 5 * eps * (::cuda::std::max(ref_av, act_av)));
      }
    }
    else
    {
      correct = correct && (h_ref[pos] == h_output[pos]);
    }
    if (!correct)
    {
      break;
    }
  }
  return correct;
}

using offsets = c2h::type_list<std::int32_t, std::uint64_t>;

CUB_TEST("Device segmented_scan works with all device interfaces",
         "[segmented][scan][device]",
         CUB_SMALL,
         full_type_list,
         offsets)
{
  using item_pair_t = typename c2h::get<0, TestType>;
  using input_t     = typename item_pair_t::input_t;
  using output_t    = typename item_pair_t::output_t;
  using offset_t    = typename c2h::get<1, TestType>;

  constexpr offset_t min_items = 2 * 1024;
  constexpr offset_t max_items = 384 * 1024;

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  const offset_t small_size  = num_items / 1024;
  const offset_t medium_size = num_items / 128;
  const offset_t large_size  = num_items / 16;

  REQUIRE(small_size > 0);

  // Range of segment sizes to generate. Ranges starting at 0 exercise empty segments.
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{0, small_size}, {medium_size, large_size}, {large_size, num_items}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << ")");

  auto device      = current_device();
  auto stream      = cuda::stream_ref{cudaStream_t{}};
  auto copy_stream = cuda::stream{device};

  // Generate input segments
  auto segment_offsets = c2h::gen_uniform_offsets_device_buffer<offset_t>(
    stream, device, C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));
  auto h_segment_offsets = make_host_buffer<offset_t>(copy_stream, device, segment_offsets.size, cuda::no_init);
  copy_to_host(copy_stream, segment_offsets.d_items.first(segment_offsets.size), h_segment_offsets);

  const auto* h_segment_offsets_ptr = h_segment_offsets.data();
  const offset_t num_segments       = static_cast<offset_t>(h_segment_offsets.size() - 1);
  auto d_offsets_it                 = segment_offsets.d_items.data();

  INFO("Num segments: " << num_segments);
  CAPTURE(c2h::type_name<input_t>(), c2h::type_name<output_t>(), c2h::type_name<offset_t>());

  // Generate input data
  const auto item_count = static_cast<std::size_t>(num_items);

  auto in_items = c2h::gen_device_buffer<input_t>(stream, device, C2H_SEED(2), item_count);
  auto h_input  = make_host_buffer<input_t>(copy_stream, device, item_count, cuda::no_init);
  copy_to_host(copy_stream, in_items, h_input);

  const auto* h_input_ptr = h_input.data();
  auto d_in_it            = in_items.data();

  auto output_vec = c2h::make_device_buffer<output_t>(stream, device, item_count, cuda::no_init);
  auto d_out_it   = output_vec.data();
#if TEST_LAUNCH == 2
  stream.sync();
#endif // TEST_LAUNCH == 2

  auto h_output = make_host_buffer<output_t>(copy_stream, device, item_count, cuda::no_init);
  auto h_ref    = make_host_buffer<output_t>(copy_stream, device, item_count, cuda::no_init);

  const auto* h_output_ptr = h_output.data();
  auto* h_ref_ptr          = h_ref.data();

  const auto verify_segment = [&](offset_t i) {
    return check_segment(h_output_ptr, h_ref_ptr, h_segment_offsets_ptr[i], h_segment_offsets_ptr[i + 1]);
  };

  SECTION("exclusive segmented scan")
  {
    using op_t = ::cuda::minimum<>;

    // check 3 offset iterators API
    device_exclusive_segmented_scan(
      d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, d_offsets_it, num_segments, op_t{}, output_t{});

    copy_to_host(copy_stream, output_vec, h_output);

    for (offset_t i = 0; i < num_segments; ++i)
    {
      compute_exclusive_scan_reference(
        h_input_ptr + h_segment_offsets_ptr[i],
        h_input_ptr + h_segment_offsets_ptr[i + 1], // NOLINT(bugprone-misplaced-widening-cast)
        h_ref_ptr + h_segment_offsets_ptr[i],
        output_t{},
        op_t{});

      const bool correct = verify_segment(i);
      REQUIRE(correct);
    }

    // check 2 offset iterators API
    device_exclusive_segmented_scan(d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, num_segments, op_t{}, output_t{});

    copy_to_host(copy_stream, output_vec, h_output);

    for (offset_t i = 0; i < num_segments; ++i)
    {
      const bool correct = verify_segment(i);
      REQUIRE(correct);
    }
  }

  SECTION("inclusive segmented scan")
  {
    using op_t      = ::cuda::std::plus<>;
    using h_accum_t = cuda::std::__accumulator_t<op_t, input_t, input_t>;

    // Scan operator
    auto scan_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // check 3 offset iterators API
    device_inclusive_segmented_scan(
      unwrap_it(d_in_it), unwrap_it(d_out_it), d_offsets_it, d_offsets_it + 1, d_offsets_it, num_segments, scan_op);

    copy_to_host(copy_stream, output_vec, h_output);

    for (offset_t i = 0; i < num_segments; ++i)
    {
      compute_inclusive_scan_reference(
        h_input_ptr + h_segment_offsets_ptr[i],
        h_input_ptr + h_segment_offsets_ptr[i + 1], // NOLINT(bugprone-misplaced-widening-cast)
        h_ref_ptr + h_segment_offsets_ptr[i],
        scan_op,
        h_accum_t{});

      const bool correct = verify_segment(i);
      REQUIRE(correct);
    }

    if constexpr (::cuda::std::is_same_v<input_t, output_t>)
    {
      cuda::copy_bytes(copy_stream, in_items, output_vec);
      copy_stream.sync();

      // check 2 iterators API for in-place scan
      device_inclusive_segmented_scan(
        unwrap_it(d_out_it), unwrap_it(d_out_it), d_offsets_it, d_offsets_it + 1, num_segments, scan_op);

      copy_to_host(copy_stream, output_vec, h_output);

      for (offset_t i = 0; i < num_segments; ++i)
      {
        const bool correct = verify_segment(i);
        REQUIRE(correct);
      }
    }
  }

  SECTION("inclusive segmented scan with init")
  {
    using op_t              = cuda::std::plus<>;
    using unwrapped_input_t = typename cuda::std::iterator_traits<decltype(unwrap_it(d_in_it))>::value_type;
    using accum_t           = cuda::std::__accumulator_t<op_t, unwrapped_input_t, unwrapped_input_t>;
    using h_accum_t         = cuda::std::__accumulator_t<op_t, input_t, input_t>;

    CAPTURE(c2h::type_name<accum_t>(), c2h::type_name<h_accum_t>());

    // Scan operator
    auto scan_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // Run test
    accum_t init_value{};
    init_default_constant(init_value);

    // check 3 offset iterators API
    device_inclusive_segmented_scan_with_init(
      unwrap_it(d_in_it),
      unwrap_it(d_out_it),
      d_offsets_it,
      d_offsets_it + 1,
      d_offsets_it,
      num_segments,
      scan_op,
      init_value);

    copy_to_host(copy_stream, output_vec, h_output);

    for (offset_t i = 0; i < num_segments; ++i)
    {
      compute_inclusive_scan_reference(
        h_input_ptr + h_segment_offsets_ptr[i],
        h_input_ptr + h_segment_offsets_ptr[i + 1], // NOLINT(bugprone-misplaced-widening-cast)
        h_ref_ptr + h_segment_offsets_ptr[i],
        scan_op,
        h_accum_t{init_value});
      // Verify result
      const bool correct = verify_segment(i);
      REQUIRE(correct);
    }

    // check 2 offset iterators API
    device_inclusive_segmented_scan_with_init(
      unwrap_it(d_in_it), unwrap_it(d_out_it), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, init_value);

    copy_to_host(copy_stream, output_vec, h_output);

    for (offset_t i = 0; i < num_segments; ++i)
    {
      // Verify result
      const bool correct = verify_segment(i);
      REQUIRE(correct);
    }
  }

#if ((TEST_TYPES == 0) || (TEST_TYPES == 1))
  SECTION("exclusive segmented sum")
  {
    using op_t = cuda::std::plus<>;

    // check 3 offset iterators API
    device_exclusive_segmented_sum(d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, d_offsets_it, num_segments);

    copy_to_host(copy_stream, output_vec, h_output);

    for (offset_t i = 0; i < num_segments; ++i)
    {
      compute_exclusive_scan_reference(
        h_input_ptr + h_segment_offsets_ptr[i],
        h_input_ptr + h_segment_offsets_ptr[i + 1], // NOLINT(bugprone-misplaced-widening-cast)
        h_ref_ptr + h_segment_offsets_ptr[i],
        output_t{},
        cuda::std::plus<>{});

      const bool correct = verify_segment(i);
      REQUIRE(correct);
    }

    // check 2 offset iterators API
    device_exclusive_segmented_sum(d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, num_segments);

    copy_to_host(copy_stream, output_vec, h_output);

    for (offset_t i = 0; i < num_segments; ++i)
    {
      const bool correct = verify_segment(i);
      REQUIRE(correct);
    }
  }

  SECTION("inclusive segmented sum")
  {
    using op_t = cuda::std::plus<>;

    // check 3 offset iterators API
    device_inclusive_segmented_sum(d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, d_offsets_it, num_segments);

    copy_to_host(copy_stream, output_vec, h_output);

    for (offset_t i = 0; i < num_segments; ++i)
    {
      compute_inclusive_scan_reference(
        h_input_ptr + h_segment_offsets_ptr[i],
        h_input_ptr + h_segment_offsets_ptr[i + 1], // NOLINT(bugprone-misplaced-widening-cast)
        h_ref_ptr + h_segment_offsets_ptr[i],
        cuda::std::plus<>{},
        output_t{});

      const bool correct = verify_segment(i);
      REQUIRE(correct);
    }

    // check 2 offset iterators API
    device_inclusive_segmented_sum(d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, num_segments);

    copy_to_host(copy_stream, output_vec, h_output);

    for (offset_t i = 0; i < num_segments; ++i)
    {
      const bool correct = verify_segment(i);
      REQUIRE(correct);
    }
  }
#endif
}
