//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Integration test: demonstrates how an algorithm consumes argument wrappers
// to make compile-time and runtime resource decisions.
// All argument types (plain values, static, dynamic, deferred) work uniformly
// through the free functions.

#include <cuda/argument>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

constexpr int shared_memory_capacity   = 256;
constexpr int default_max_segment_size = 1024;

enum class algorithm_variant
{
  shared_memory,
  global_memory
};

// Static scaling: choose algorithm variant at compile time.
template <class _SegSizeArg>
TEST_FUNC constexpr algorithm_variant select_variant(_SegSizeArg)
{
  if constexpr (cuda::argument::__traits<_SegSizeArg>::max <= shared_memory_capacity)
  {
    return algorithm_variant::shared_memory;
  }
  else
  {
    return algorithm_variant::global_memory;
  }
}

// Dynamic scaling: compute buffer size at runtime, clamped to default.
template <class _SegSizeArg>
TEST_FUNC constexpr int compute_buffer_size(_SegSizeArg __seg_size, int __num_segments)
{
  auto __max = cuda::std::min(default_max_segment_size, static_cast<int>(cuda::argument::__max(__seg_size)));
  return __max * __num_segments;
}

// Process: use the actual unwrapped value.
template <class _SegSizeArg>
TEST_FUNC constexpr int process_segments(_SegSizeArg __seg_size)
{
  const auto& __val = cuda::argument::__unwrap(__seg_size);

  if constexpr (cuda::argument::__is_single_value_v<
                  cuda::std::remove_cv_t<cuda::std::remove_reference_t<decltype(__val)>>>)
  {
    return static_cast<int>(__val);
  }
  else
  {
    int __total = 0;
    for (size_t __i = 0; __i < __val.size(); ++__i)
    {
      __total += static_cast<int>(__val[__i]);
    }
    return __total;
  }
}

TEST_FUNC constexpr bool test()
{
  // Plain scalar: no bounds, global memory, buffer clamped to default
  {
    static_assert(select_variant(100) == algorithm_variant::global_memory);
    assert(compute_buffer_size(100, 4) == default_max_segment_size * 4);
    assert(process_segments(100) == 100);
  }

  // Plain span: per-segment, no bounds, global memory
  {
    int sizes[3] = {64, 128, 96};
    auto seg     = cuda::std::span<int>{sizes, 3};
    assert(select_variant(seg) == algorithm_variant::global_memory);
    assert(compute_buffer_size(seg, 3) == default_max_segment_size * 3);
    assert(process_segments(seg) == 64 + 128 + 96);
  }

  // static_argument: scalar, fits in shared memory, buffer = value
  {
    constexpr auto seg_size = cuda::argument::__constant<128>{};
    static_assert(select_variant(seg_size) == algorithm_variant::shared_memory);
    assert(compute_buffer_size(seg_size, 4) == 128 * 4);
    assert(process_segments(seg_size) == 128);
  }

#if defined(__cpp_nontype_template_args) && __cpp_nontype_template_args >= 201911L
  // static_argument: array, max fits in shared memory
  {
    constexpr auto seg_sizes = cuda::argument::__constant<cuda::std::array{64, 128, 256}>{};
    static_assert(select_variant(seg_sizes) == algorithm_variant::shared_memory);
    assert(compute_buffer_size(seg_sizes, 3) == 256 * 3);
    assert(process_segments(seg_sizes) == 64 + 128 + 256);
  }

  // static_argument: array, max exceeds shared memory, buffer clamped
  {
    constexpr auto seg_sizes = cuda::argument::__constant<cuda::std::array{64, 128, 512}>{};
    static_assert(select_variant(seg_sizes) == algorithm_variant::global_memory);
    assert(compute_buffer_size(seg_sizes, 3) == 512 * 3);
    assert(process_segments(seg_sizes) == 64 + 128 + 512);
  }
#endif // _CCCL_STD_VER >= 2020

  // dynamic_argument: tight static bounds, shared memory, buffer = static max
  {
    constexpr auto seg_size = cuda::argument::__dynamic{100, cuda::argument::__bounds<1, 256>()};
    static_assert(select_variant(seg_size) == algorithm_variant::shared_memory);
    assert(compute_buffer_size(seg_size, 4) == 256 * 4);
    assert(process_segments(seg_size) == 100);
  }

  // dynamic_argument: wide static bounds, global memory, buffer clamped to default
  {
    constexpr auto seg_size = cuda::argument::__dynamic{100, cuda::argument::__bounds<1, 4096>()};
    static_assert(select_variant(seg_size) == algorithm_variant::global_memory);
    assert(compute_buffer_size(seg_size, 4) == default_max_segment_size * 4);
    assert(process_segments(seg_size) == 100);
  }

  // dynamic_argument: no bounds, global memory, buffer clamped
  {
    constexpr auto seg_size = cuda::argument::__dynamic{100};
    static_assert(select_variant(seg_size) == algorithm_variant::global_memory);
    assert(compute_buffer_size(seg_size, 4) == default_max_segment_size * 4);
    assert(process_segments(seg_size) == 100);
  }

  // dynamic_argument: per-segment span with runtime bounds only
  {
    int sizes[3]   = {64, 128, 96};
    auto seg_sizes = cuda::argument::__dynamic{cuda::std::span<int>{sizes, 3}, cuda::argument::__bounds(1, 200)};
    assert(select_variant(seg_sizes) == algorithm_variant::global_memory);
    assert(compute_buffer_size(seg_sizes, 3) == 200 * 3);
    assert(process_segments(seg_sizes) == 64 + 128 + 96);
  }

  // dynamic_argument: per-segment span with both bounds
  {
    int sizes[3]   = {64, 128, 96};
    auto seg_sizes = cuda::argument::__dynamic{
      cuda::std::span<int>{sizes, 3}, cuda::argument::__bounds<1, 256>(), cuda::argument::__bounds(1, 200)};
    static_assert(cuda::argument::__traits<decltype(seg_sizes)>::max <= shared_memory_capacity);
    assert(select_variant(seg_sizes) == algorithm_variant::shared_memory);
    assert(compute_buffer_size(seg_sizes, 3) == 200 * 3);
    assert(process_segments(seg_sizes) == 64 + 128 + 96);
  }

  // deferred_argument: uniform, bounds for decisions only
  {
    int val       = 100;
    auto seg_size = cuda::argument::__deferred{
      cuda::std::span<int, 1>{&val, 1}, cuda::argument::__bounds<1, 256>(), cuda::argument::__bounds(1, 200)};
    static_assert(cuda::argument::__traits<decltype(seg_size)>::max <= shared_memory_capacity);
    assert(select_variant(seg_size) == algorithm_variant::shared_memory);
    assert(compute_buffer_size(seg_size, 4) == 200 * 4);
  }

  // --- Floating point cases ---

  // Plain float: no bounds
  {
    static_assert(select_variant(1.0f) == algorithm_variant::global_memory);
    assert(process_segments(1.0f) == 1);
  }

#if defined(__cpp_nontype_template_args) && __cpp_nontype_template_args >= 201911L
  // static_argument float (float NTTPs require C++20)
  {
    constexpr auto seg_size = cuda::argument::__constant<128.0f>{};
    static_assert(select_variant(seg_size) == algorithm_variant::shared_memory);
    assert(process_segments(seg_size) == 128);
  }

  // dynamic_argument float with static bounds
  {
    constexpr auto seg_size = cuda::argument::__dynamic{100.0f, cuda::argument::__bounds<1.0f, 256.0f>()};
    static_assert(select_variant(seg_size) == algorithm_variant::shared_memory);
    assert(process_segments(seg_size) == 100);
  }
#endif // _CCCL_STD_VER >= 2020

  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
