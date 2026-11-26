// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This benchmark checks that cub::DeviceScan unwraps fancy iterators to the underlying raw pointers, so the algorithm's
// fast paths (vectorized loads, bulk copy) are still taken. It is a proof of the optimization, not a tuning or sweeping
// benchmark. It is the reproducer from https://github.com/NVIDIA/cccl/issues/11840, mirroring cuDF's logical stack: a
// scan of 0/1 reset flags into segment ids. The two benchmarks move the same number of bytes and should perform the
// same:
// * reset_segment_scan_pointers scans a materialized buffer of reset flags into a raw output pointer.
// * reset_segment_scan_fancy_iterators scans a transform_iterator over the raw symbols into a transform_output_iterator
//   over the raw output pointer.

#include <cub/device/device_scan.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <stdexcept>
#include <string>

#include <nvbench_helper.cuh>

// mirrors cudf::io::fst::stack_op_type
enum class stack_op_type : ::cuda::std::int32_t
{
  READ,
  PUSH,
  POP,
  RESET
};

// types as instantiated in cuDF
using stack_symbol_t  = char; // cudf::io::json::StackSymbolT
using stack_segment_t = ::cuda::std::uint8_t; // logical_stack.cuh: StackSegmentT

template <bool UseFancyIterators>
void reset_segment_scan(nvbench::state& state)
try
{
  // number of stack operations (brackets/braces/newlines outside of quotes)
  const auto num_symbols  = static_cast<::cuda::std::size_t>(state.get_int64("NumSymbols"));
  const auto reset_period = static_cast<::cuda::std::size_t>(state.get_int64("ResetPeriod"));

  // sparse stack-op sequence: balanced '{' / '[' pushes and pops, '\n' every reset_period symbols
  thrust::device_vector<stack_symbol_t> stack_ops(num_symbols, thrust::no_init);
  thrust::tabulate(
    stack_ops.begin(), stack_ops.end(), [reset_period] __device__(::cuda::std::size_t i) -> stack_symbol_t {
      constexpr char pattern[] = {'{', '[', ']', '}'};
      return (i % reset_period == 0) ? '\n' : pattern[i % 4];
    });

  // mirrors JSONWithRecoveryToStackOp (nested_json_gpu.cu)
  auto symbol_to_stack_op = [] __host__ __device__(const stack_symbol_t& stack_symbol) -> stack_op_type {
    switch (stack_symbol)
    {
      case '{':
      case '[':
        return stack_op_type::PUSH;
      case '}':
      case ']':
        return stack_op_type::POP;
      case '\n':
        return stack_op_type::RESET;
      default:
        return stack_op_type::READ;
    }
  };

  // mirrors detail::NewlineToResetStackSegmentOp<JSONWithRecoveryToStackOp>
  auto newline_to_reset_segment = [symbol_to_stack_op] __host__ __device__(const stack_symbol_t& stack_symbol)
    -> ::cuda::std::uint32_t {
      // every reset operation marks the beginning of a new segment
      return symbol_to_stack_op(stack_symbol) == stack_op_type::RESET ? 1 : 0;
    };

  // mirrors detail::ModToTargetTypeOpT<StackSegmentT>, instantiated with T = uint32_t
  auto mod_to_target_type = [] __host__ __device__(const ::cuda::std::uint32_t& val) -> stack_segment_t {
    return static_cast<stack_segment_t>(
      val % (static_cast<::cuda::std::uint32_t>(::cuda::std::numeric_limits<stack_segment_t>::max()) + 1));
  };

  thrust::device_vector<stack_segment_t> key_segments(num_symbols);

  state.add_element_count(num_symbols);
  state.add_global_memory_reads<stack_symbol_t>(num_symbols);
  state.add_global_memory_writes<stack_segment_t>(num_symbols);

  caching_allocator_t alloc;
  auto scan = [&](auto in, auto out) {
    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceScan::InclusiveSum, "InclusiveSum failed", in, out, num_symbols, cub_bench_env(alloc, launch));
    });
  };

  if constexpr (UseFancyIterators)
  {
    scan(cuda::transform_iterator(thrust::raw_pointer_cast(stack_ops.data()), newline_to_reset_segment),
         cuda::make_transform_output_iterator(thrust::raw_pointer_cast(key_segments.data()), mod_to_target_type));
  }
  else
  {
    thrust::device_vector<stack_segment_t> reset_flags(num_symbols);
    thrust::transform(stack_ops.begin(), stack_ops.end(), reset_flags.begin(), newline_to_reset_segment);
    scan(thrust::raw_pointer_cast(reset_flags.data()), thrust::raw_pointer_cast(key_segments.data()));
  }

  // verify the result: the segment id increments after every reset symbol
  const auto mismatches = thrust::count_if(
    thrust::counting_iterator<::cuda::std::size_t>{0},
    thrust::counting_iterator<::cuda::std::size_t>{num_symbols},
    [segments = thrust::raw_pointer_cast(key_segments.data()), reset_period] __device__(::cuda::std::size_t i) {
      return segments[i] != static_cast<stack_segment_t>(((i / reset_period) + 1) % 256);
    });
  if (mismatches != 0)
  {
    throw std::runtime_error("Verification failed: " + std::to_string(mismatches) + " mismatches");
  }
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

// NVBENCH_BENCH cannot take a template-id
static void reset_segment_scan_pointers(nvbench::state& state)
{
  reset_segment_scan<false>(state);
}

NVBENCH_BENCH(reset_segment_scan_pointers)
  .set_name("reset_segment_scan_pointers")
  .add_int64_power_of_two_axis("NumSymbols", {20, 24, 28, 31})
  .add_int64_axis("ResetPeriod", {1024});

static void reset_segment_scan_fancy_iterators(nvbench::state& state)
{
  reset_segment_scan<true>(state);
}

NVBENCH_BENCH(reset_segment_scan_fancy_iterators)
  .set_name("reset_segment_scan_fancy_iterators")
  .add_int64_power_of_two_axis("NumSymbols", {20, 24, 28, 31})
  .add_int64_axis("ResetPeriod", {1024});
