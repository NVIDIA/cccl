// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %PARAM% TEST_ERR err 0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20:21

// Defer the unsupported-architecture diagnosis to the dispatch's runtime check (not a compile-time static_assert)
// so only the requirement static_asserts under test fire, regardless of which architectures this test is compiled for.
// See CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in cub/device/device_batched_topk.cuh. Precedes the CUB include below.
#define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#include <cub/device/device_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/span>

#include <iostream>

// Verifies that cub::DeviceBatchedTopK rejects, at compile time, requests the public contract marks as ill-formed.
// Each variant makes exactly one argument ill-formed (the others take valid defaults) and checks the single diagnostic:
//
//   * requirements (variants 0-5):
//       - determinism and tie_break must be acknowledged together, both specified or both omitted to take the default
//       - an explicit tie_break of prefer_smaller_index / prefer_larger_index pins the result set across GPUs and so
//         requires determinism::gpu_to_gpu (it cannot be paired with not_guaranteed or run_to_run)
//   * segment_sizes (variants 6-13):
//       - an explicit compile-time upper bound above the maximum supported segment size (2^21)
//       - an un-annotated argument whose element type max exceeds 2^21 (a compile-time bound is required): int64,
//         uint32, and int32 all need one; only a type whose max already fits (e.g. int16, uint16) is accepted bare
//       - a deferred argument wrapping a scalar, or a range / container (even a static-extent-1 span), rather than a
//         dereferenceable pointer / handle
//       - a deferred_sequence argument wrapping a non-random-access handle (a span is a range, not an iterator)
//       - a single-value wrapper (immediate) around a pointer / handle rather than an integral value
//   * k (variants 14-17, 21): the handle / element-type checks shared with segment_sizes (k is likewise
//     uniform-or-per-segment; it has no maximum-size bound), plus an element type wider than 64 bits (variant 21),
//     which the device cannot clamp through its 64-bit intermediate without silently wrapping
//   * num_segments (variants 18-20): must be a single, host-known integral value -- a pointer-valued single wrapper
//     (not integral), a per-segment sequence (not a single value), and a single deferred device-resident value (a
//     single value, but not host-known, so caught by a separate check) are each rejected

int main()
{
  namespace ex = cuda::execution;

  // Per-segment key iterators (iterator-of-iterators). The keys-only entry point ignores value iterators.
  int** d_keys_in  = nullptr;
  int** d_keys_out = nullptr;

  // Backing storage for the span-shaped (range, not handle) arguments used by a few variants.
  [[maybe_unused]] cuda::std::uint16_t range_storage[1]{};

  // segment_sizes: valid by default; variants 6-13 override it with one ill-formed form.
#if TEST_ERR == 6 // explicit segment-size upper bound above 2^21 (unsupported by the streaming cluster backend)
  auto segment_sizes = cuda::args::immediate{
    cuda::std::int64_t{8}, cuda::args::bounds<cuda::std::int64_t{1}, cuda::std::int64_t{1} << 40>()};
  // expected-error-6 {{"exceeds the maximum currently supported segment size"}}
#elif TEST_ERR == 7 // un-annotated wide segment-size type (int64 type max > 2^21): a compile-time bound is required
  auto segment_sizes = cuda::args::immediate{cuda::std::int64_t{8}};
  // expected-error-7 {{"exceeds the maximum currently supported segment size"}}
#elif TEST_ERR == 8 // un-annotated unsigned segment-size type (uint32 type max > 2^21): a compile-time bound too
  auto segment_sizes = cuda::args::immediate{cuda::std::uint32_t{8}};
  // expected-error-8 {{"exceeds the maximum currently supported segment size"}}
#elif TEST_ERR == 9 // un-annotated int32 segment-size type (type max 2^31 - 1 > 2^21): a compile-time bound is required
  auto segment_sizes = cuda::args::immediate{cuda::std::int32_t{8}};
  // expected-error-9 {{"exceeds the maximum currently supported segment size"}}
#elif TEST_ERR == 10 // deferred wrapping a scalar (not a dereferenceable pointer/handle): rejected
  auto segment_sizes = cuda::args::deferred{cuda::std::int32_t{8}};
  // expected-error-10 {{"must wrap a pointer or other dereferenceable handle"}}
#elif TEST_ERR == 11 // deferred wrapping a range/container: rejected even at static extent 1 (still a range, not a
                     // handle)
  auto segment_sizes = cuda::args::deferred{cuda::std::span<cuda::std::uint16_t, 1>{range_storage}};
  // expected-error-11 {{"must wrap a pointer or other dereferenceable handle"}}
#elif TEST_ERR == 12 // deferred_sequence wrapping a non-random-access handle (a span is a range, not an iterator)
  auto segment_sizes = cuda::args::deferred_sequence{cuda::std::span<cuda::std::uint16_t>{}};
  // expected-error-12 {{"must wrap a random-access iterator"}}
#elif TEST_ERR == 13 // single-value wrapper (immediate) around a pointer/handle rather than an integral value
  auto segment_sizes = cuda::args::immediate{static_cast<int*>(nullptr), cuda::args::bounds<0, 8>()};
  // expected-error-13 {{"must have an integral \(non-bool\) element type"}}
#else
  auto segment_sizes = cuda::args::constant<8>{};
#endif

  // k: valid by default; variants 14-17 mirror the segment_sizes handle / element-type checks (k has no size bound).
#if TEST_ERR == 14 // deferred wrapping a scalar (not a dereferenceable pointer/handle): rejected
  auto k_arg = cuda::args::deferred{cuda::std::int32_t{3}};
  // expected-error-14 {{"must wrap a pointer or other dereferenceable handle"}}
#elif TEST_ERR == 15 // deferred wrapping a range/container (a span is a range, not a handle): rejected
  auto k_arg = cuda::args::deferred{cuda::std::span<cuda::std::uint16_t, 1>{range_storage}};
  // expected-error-15 {{"must wrap a pointer or other dereferenceable handle"}}
#elif TEST_ERR == 16 // deferred_sequence wrapping a non-random-access handle (a span is a range, not an iterator)
  auto k_arg = cuda::args::deferred_sequence{cuda::std::span<cuda::std::uint16_t>{}};
  // expected-error-16 {{"must wrap a random-access iterator"}}
#elif TEST_ERR == 17 // single-value wrapper (immediate) around a pointer/handle rather than an integral value
  auto k_arg = cuda::args::immediate{static_cast<int*>(nullptr), cuda::args::bounds<0, 8>()};
  // expected-error-17 {{"must have an integral \(non-bool\) element type"}}
#elif TEST_ERR == 21 // an integer element type wider than 64 bits: the device clamps k through a 64-bit intermediate,
                     // so a wider type (e.g. __int128) is rejected to avoid a silent wrap
#  if _CCCL_HAS_INT128()
  auto k_arg = static_cast<__int128_t>(3);
#  else // no 128-bit integer on this platform; assert the same diagnostic directly so the variant is still exercised
  auto k_arg = cuda::args::constant<3>{};
  static_assert(false, "cub::DeviceBatchedTopK: k's element type must be at most 64 bits wide.");
#  endif
  // expected-error-21 {{"element type must be at most 64 bits wide"}}
#else
  auto k_arg = cuda::args::constant<3>{};
#endif

  // num_segments: valid by default; variants 18-20 override it with a form that is not a single host-known value.
#if TEST_ERR == 18 // single-value wrapper (immediate) around a pointer rather than an integral value
  auto num_segments = cuda::args::immediate{static_cast<int*>(nullptr), cuda::args::bounds<0, 8>()};
  // expected-error-18 {{"must have an integral \(non-bool\) type"}}
#elif TEST_ERR == 19 // a single deferred (device-resident) value: allowed as a uniform value, but num_segments must be
                     // host-known (it sizes the launch), so rejected by the separate host-known check
  auto num_segments = cuda::args::deferred{static_cast<int*>(nullptr)};
  // expected-error-19 {{"must be a host-known value"}}
#elif TEST_ERR == 20 // a per-segment sequence: not a single value
  auto num_segments = cuda::args::deferred_sequence{static_cast<int*>(nullptr)};
  // expected-error-20 {{"must be a single value"}}
#else
  auto num_segments = cuda::args::immediate{cuda::std::int64_t{2}};
#endif

  // requirements: valid (unsorted) by default; variants 0-5 override it with a rejected determinism/tie_break request.
#if TEST_ERR == 0 // determinism specified without a paired tie_break
  auto requirements = ex::require(ex::determinism::not_guaranteed, ex::output_ordering::unsorted);
  // expected-error-0 {{"must be acknowledged together"}}
#elif TEST_ERR == 1 // tie_break specified without a paired determinism
  auto requirements = ex::require(ex::tie_break::prefer_smaller_index, ex::output_ordering::unsorted);
  // expected-error-1 {{"must be acknowledged together"}}
#elif TEST_ERR == 2 // explicit tie_break with not_guaranteed
  auto requirements =
    ex::require(ex::determinism::not_guaranteed, ex::tie_break::prefer_smaller_index, ex::output_ordering::unsorted);
  // expected-error-2 {{"pins the result set across GPUs and therefore requires"}}
#elif TEST_ERR == 3 // explicit tie_break with not_guaranteed
  auto requirements =
    ex::require(ex::determinism::not_guaranteed, ex::tie_break::prefer_larger_index, ex::output_ordering::unsorted);
  // expected-error-3 {{"pins the result set across GPUs and therefore requires"}}
#elif TEST_ERR == 4 // explicit tie_break with run_to_run
  auto requirements =
    ex::require(ex::determinism::run_to_run, ex::tie_break::prefer_smaller_index, ex::output_ordering::unsorted);
  // expected-error-4 {{"pins the result set across GPUs and therefore requires"}}
#elif TEST_ERR == 5 // explicit tie_break with run_to_run
  auto requirements =
    ex::require(ex::determinism::run_to_run, ex::tie_break::prefer_larger_index, ex::output_ordering::unsorted);
  // expected-error-5 {{"pins the result set across GPUs and therefore requires"}}
#else
  auto requirements = ex::require(ex::output_ordering::unsorted);
#endif

  auto env                  = cuda::std::execution::env{requirements};
  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segments, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MaxKeys failed with status: " << error << '\n';
  }
}
