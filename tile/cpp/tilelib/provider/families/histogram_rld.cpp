// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/histogram_rld.h"

#include <climits>
#include <utility>
#include <vector>

#include "provider/source_common.h"

namespace cuda::coop::tile::detail::provider
{
namespace
{
bool fail(std::string message, std::string* error)
{
  if (error)
  {
    *error = std::move(message);
  }
  return false;
}

constexpr char histogram_template[] = R"CUDA(
#include <cub/block/block_histogram.cuh>

namespace {
#ifndef CUDA_COOP_TILE_BLOCK_THREADS
#define CUDA_COOP_TILE_BLOCK_THREADS 128
#endif
constexpr int kBlockThreads = CUDA_COOP_TILE_BLOCK_THREADS;
#ifndef CUDA_COOP_TILE_ITEMS_PER_THREAD
#define CUDA_COOP_TILE_ITEMS_PER_THREAD 4
#endif
constexpr int kItemsPerThread = CUDA_COOP_TILE_ITEMS_PER_THREAD;
constexpr int kBins = %BINS%;
static_assert(kBins % kBlockThreads == 0,
              "histogram bins must be divisible by block threads");
constexpr int kBinsPerThread = kBins / kBlockThreads;

using SampleT = unsigned char;
using CounterT = int;
using BlockHistogramT =
    cub::BlockHistogram<SampleT, kBlockThreads, kItemsPerThread, kBins,
                        cub::BLOCK_HISTO_ATOMIC>;
struct ScratchStorage {
    typename BlockHistogramT::TempStorage temp_storage;
    CounterT smem_hist[kBins];
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockHistogram scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockHistogram alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    const SampleT* input_rmem,
    CounterT* output_rmem,
    void* scratch_smem)
{
    SampleT samples[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        samples[i] = input_rmem[i];
    }

    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);

    BlockHistogramT(scratch.temp_storage).Histogram(samples, scratch.smem_hist);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < kBinsPerThread; ++i) {
        output_rmem[i] = scratch.smem_hist[%OUTPUT_INDEX%];
    }
    __syncthreads();
}
)CUDA";

constexpr char run_length_decode_template[] = R"CUDA(
%INCLUDES%

namespace {
#ifndef CUDA_COOP_TILE_BLOCK_THREADS
#define CUDA_COOP_TILE_BLOCK_THREADS 128
#endif
constexpr int kBlockThreads = CUDA_COOP_TILE_BLOCK_THREADS;
#ifndef CUDA_COOP_TILE_ITEMS_PER_THREAD
#define CUDA_COOP_TILE_ITEMS_PER_THREAD 2
#endif
constexpr int kRunsPerThread = CUDA_COOP_TILE_ITEMS_PER_THREAD;
constexpr int kDecodedItemsPerThread = %DECODED_ITEMS_PER_THREAD%;

using ValueT = %VALUE_T%;
using LengthT = %LENGTH_T%;
using BlockRunLengthDecodeT =
    cub::BlockRunLengthDecode<ValueT,
                              kBlockThreads,
                              kRunsPerThread,
                              kDecodedItemsPerThread,
                              LengthT>;
struct ScratchStorage {
    typename BlockRunLengthDecodeT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockRunLengthDecode scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockRunLengthDecode alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    const ValueT* run_values_rmem,
    const LengthT* run_lengths_rmem,
    LengthT runtime_decoded_window_offset,
    ValueT* decoded_items_rmem,
%RELATIVE_OFFSET_PARAMETER%
    void* scratch_smem)
{
    ValueT run_values[kRunsPerThread];
    LengthT run_lengths[kRunsPerThread];
#pragma unroll
    for (int i = 0; i < kRunsPerThread; ++i) {
        run_values[i] = run_values_rmem[i];
        run_lengths[i] = run_lengths_rmem[i];
    }

    ValueT decoded_items[kDecodedItemsPerThread];
%RELATIVE_OFFSET_DECL%
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    LengthT total_decoded_size = 0;
    BlockRunLengthDecodeT decoder(
        scratch.temp_storage, run_values, run_lengths, total_decoded_size);
    const LengthT safe_decoded_window_offset =
        runtime_decoded_window_offset < total_decoded_size
            ? runtime_decoded_window_offset
            : static_cast<LengthT>(0);
%DECODE_CALL%

    // CUB's final reads consume its TempStorage. Synchronize before the
    // compiler may reuse the provider-owned scratch allocation.
    __syncthreads();

    const LengthT remaining =
        runtime_decoded_window_offset < total_decoded_size
            ? total_decoded_size - runtime_decoded_window_offset
            : static_cast<LengthT>(0);
    const unsigned long long thread_base =
        static_cast<unsigned long long>(threadIdx.x) * kDecodedItemsPerThread;
#pragma unroll
    for (int i = 0; i < kDecodedItemsPerThread; ++i) {
        const unsigned long long local_index =
            thread_base + static_cast<unsigned long long>(i);
        const bool valid =
            local_index < static_cast<unsigned long long>(remaining);
        decoded_items_rmem[i] = valid ? decoded_items[i] : %VALUE_ZERO%;
%RELATIVE_OFFSET_PROJECTION%
    }
}
)CUDA";
} // namespace

// Emit uint8 sampling with int32 counters and a blocked output-bin fragment.
// The complete shared histogram is wrapper scratch in addition to CUB's own
// TempStorage. Bin divisibility ensures each worker returns an integral
// number of consecutive counters.
bool emit_histogram_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null Histogram source destination", error);
  }
  const auto* const spec = std::get_if<HistogramSpec>(&request.primitive());
  if (!spec)
  {
    return fail("Histogram emitter received a non-Histogram PrimitiveSpec", error);
  }
  if (spec->bins <= 0)
  {
    return fail("invalid Histogram bins=" + std::to_string(spec->bins), error);
  }
  if (spec->fixed_256 && spec->bins != 256)
  {
    return fail("Histogram256 requires bins=256", error);
  }
  if (spec->bins > 256)
  {
    return fail("Histogram bins must be <= 256", error);
  }
  if (!spec->dtype.empty() && spec->dtype != "u8")
  {
    return fail("Histogram requires dtype=u8, got " + spec->dtype, error);
  }
  if (request.topology().block_threads <= 0 || (spec->bins % request.topology().block_threads) != 0)
  {
    return fail("Histogram bins=" + std::to_string(spec->bins)
                  + " must be divisible by block_threads=" + std::to_string(request.topology().block_threads),
                error);
  }

  const std::string& entry_name = requested_entry_name;
  std::string result            = histogram_template;
  result                        = replace_all(std::move(result), "%BINS%", std::to_string(spec->bins));
  result                        = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  result  = replace_all(std::move(result), "%OUTPUT_INDEX%", "static_cast<int>(threadIdx.x) * kBinsPerThread + i");
  *source = std::move(result);
  return true;
}

// Emit a fixed-size decoded window, optionally with relative run offsets.
// Run and decoded per-thread extents are distinct specialization quantities.
// The runtime window offset remains an operand. The wrapper uses a safe CUB
// window for out-of-range requests, then writes zeros and all-ones offset
// sentinels for logical positions beyond the decoded sequence.
bool emit_run_length_decode_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null RunLengthDecode source destination", error);
  }
  const auto* const spec = std::get_if<RunLengthDecodeSpec>(&request.primitive());
  if (!spec)
  {
    return fail("RunLengthDecode emitter received a non-RunLengthDecode PrimitiveSpec", error);
  }

  std::string value_cpp_type;
  std::string value_extra_include;
  std::string value_zero_expression;
  std::string length_cpp_type;
  std::string length_extra_include;
  std::string unused_length_zero_expression;
  if (!decode_dtype_suffix(spec->value_dtype, &value_cpp_type, &value_extra_include, &value_zero_expression)
      || !decode_dtype_suffix(
        spec->length_dtype, &length_cpp_type, &length_extra_include, &unused_length_zero_expression)
      || (spec->length_dtype != "u32" && spec->length_dtype != "u64"))
  {
    return fail("unsupported RunLengthDecode dtypes value=" + spec->value_dtype + " length=" + spec->length_dtype,
                error);
  }
  if (request.topology().decoded_items_per_thread <= 0)
  {
    return fail(
      "invalid RunLengthDecode decoded_items_per_thread=" + std::to_string(request.topology().decoded_items_per_thread),
      error);
  }

  const std::string& entry_name           = requested_entry_name;
  const std::vector<std::string> includes = {
    value_extra_include,
    length_extra_include,
    "cuda_runtime.h",
    "cub/block/block_run_length_decode.cuh",
  };

  std::string result = run_length_decode_template;
  result             = replace_all(std::move(result), "%INCLUDES%", build_include_block(includes));
  result             = replace_all(std::move(result), "%VALUE_T%", value_cpp_type);
  result             = replace_all(std::move(result), "%LENGTH_T%", length_cpp_type);
  result             = replace_all(
    std::move(result), "%DECODED_ITEMS_PER_THREAD%", std::to_string(request.topology().decoded_items_per_thread));
  result = replace_all(std::move(result),
                       "%RELATIVE_OFFSET_PARAMETER%",
                       spec->with_offsets ? "    LengthT* relative_offsets_rmem,\n" : "");
  result = replace_all(std::move(result),
                       "%RELATIVE_OFFSET_DECL%",
                       spec->with_offsets ? "    LengthT relative_offsets[kDecodedItemsPerThread];\n" : "");
  result = replace_all(
    std::move(result),
    "%DECODE_CALL%",
    spec->with_offsets
      ? "    decoder.RunLengthDecode(\n"
        "        decoded_items, relative_offsets, safe_decoded_window_offset);"
      : "    decoder.RunLengthDecode(decoded_items, safe_decoded_window_offset);");
  result = replace_all(
    std::move(result),
    "%RELATIVE_OFFSET_PROJECTION%",
    spec->with_offsets
      ? "        relative_offsets_rmem[i] =\n"
        "            valid ? relative_offsets[i]\n"
        "                  : static_cast<LengthT>(~static_cast<LengthT>(0));"
      : "");
  result  = replace_all(std::move(result), "%VALUE_ZERO%", value_zero_expression);
  result  = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  *source = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
