// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/radix_rank.h"

#include <utility>

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

constexpr char radix_rank_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cuda/std/type_traits>

namespace {
#ifndef CUDA_COOP_TILE_BLOCK_THREADS
#define CUDA_COOP_TILE_BLOCK_THREADS 128
#endif
constexpr int kBlockThreads = CUDA_COOP_TILE_BLOCK_THREADS;
#ifndef CUDA_COOP_TILE_ITEMS_PER_THREAD
#define CUDA_COOP_TILE_ITEMS_PER_THREAD 1
#endif
constexpr int kItemsPerThread = CUDA_COOP_TILE_ITEMS_PER_THREAD;
constexpr int kRadixBits = %RADIX_BITS%;
constexpr bool kDescending = %DESCENDING%;
constexpr bool kMemoizeOuterScan = %MEMOIZE_OUTER_SCAN%;
constexpr cub::BlockScanAlgorithm kInnerScanAlgorithm = %INNER_SCAN_ALGO%;
constexpr cudaSharedMemConfig kSmemConfig = %SMEM_CONFIG%;
using KeyT = %KEY_T%;
using UnsignedKeyT = ::cuda::std::make_unsigned_t<KeyT>;
using BlockRadixRankT =
    cub::BlockRadixRank<
        kBlockThreads,
        kRadixBits,
        kDescending,
        kMemoizeOuterScan,
        kInnerScanAlgorithm,
        kSmemConfig>;
struct ScratchStorage {
    typename BlockRadixRankT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockRadixRank scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockRadixRank alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    KeyT* input_rmem,
%BIT_CONTROL_PARAMETERS%
    int* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    const long long runtime_radix_bits_wide =
        static_cast<long long>(runtime_end_bit) - static_cast<long long>(runtime_begin_bit);
    if (runtime_begin_bit < 0 || runtime_begin_bit >= %KEY_BITS% ||
        runtime_end_bit <= runtime_begin_bit || runtime_end_bit > %KEY_BITS% ||
        runtime_radix_bits_wide > kRadixBits) {
        asm volatile("trap;");
        return;
    }
    const int runtime_radix_bits = static_cast<int>(runtime_radix_bits_wide);

    UnsignedKeyT thread_keys[kItemsPerThread];
    int thread_ranks[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_keys[i] = static_cast<UnsignedKeyT>(input_rmem[i]);
        if constexpr (::cuda::std::is_signed_v<KeyT>) {
            thread_keys[i] ^= static_cast<UnsignedKeyT>(
                UnsignedKeyT{1} << (sizeof(KeyT) * 8 - 1));
        }
    }

    cub::BFEDigitExtractor<UnsignedKeyT> extractor(runtime_begin_bit, runtime_radix_bits);
    BlockRadixRankT(scratch.temp_storage).RankKeys(thread_keys, thread_ranks, extractor);

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = thread_ranks[i];
    }
    __syncthreads();
}
)CUDA";
} // namespace

// Emit one ranking pass whose compiled radix width bounds the runtime interval.
// The generated call accepts begin/end operands, checks them against both key
// width and compiled capacity, and writes int32 ranks. Runtime bounds must not
// be substituted into the specialization's identity.
bool emit_radix_rank_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null RadixRank source destination", error);
  }
  const auto* const spec = std::get_if<RadixRankSpec>(&request.primitive());
  if (!spec)
  {
    return fail("RadixRank emitter received a non-RadixRank PrimitiveSpec", error);
  }
  if (!is_integer_dtype_suffix(spec->dtype))
  {
    return fail("RadixRank requires integer dtype=" + spec->dtype, error);
  }

  std::string cpp_type;
  std::string extra_include;
  std::string unused_zero_expression;
  if (!decode_dtype_suffix(spec->dtype, &cpp_type, &extra_include, &unused_zero_expression))
  {
    return fail("unsupported RadixRank dtype=" + spec->dtype, error);
  }

  const int key_bits               = bitwidth_for_dtype_suffix(spec->dtype);
  const std::string inner_scan_cpp = resolve_scan_algorithm_token(spec->inner_scan_algorithm);
  const std::string smem_cpp       = resolve_smem_config_token(spec->smem_config);
  const int radix_bits             = spec->radix_bits;
  if (inner_scan_cpp.empty() || smem_cpp.empty())
  {
    return fail("unsupported RadixRank tuning inner_scan_algorithm=" + spec->inner_scan_algorithm
                  + " smem_config=" + spec->smem_config,
                error);
  }
  if (radix_bits <= 0 || radix_bits > 16)
  {
    return fail("invalid RadixRank radix_bits=" + std::to_string(radix_bits), error);
  }

  const std::string& entry_name            = requested_entry_name;
  const std::string bit_control_parameters = "    int runtime_begin_bit,\n    int runtime_end_bit,\n";
  std::string result                       = radix_rank_template;
  result                                   = replace_all(
    std::move(result), "%EXTRA_INCLUDE%", extra_include.empty() ? "" : ("#include <" + extra_include + ">\n"));
  result  = replace_all(std::move(result), "%KEY_T%", cpp_type);
  result  = replace_all(std::move(result), "%RADIX_BITS%", std::to_string(radix_bits));
  result  = replace_all(std::move(result), "%DESCENDING%", spec->descending ? "true" : "false");
  result  = replace_all(std::move(result), "%MEMOIZE_OUTER_SCAN%", spec->memoize_outer_scan ? "true" : "false");
  result  = replace_all(std::move(result), "%INNER_SCAN_ALGO%", inner_scan_cpp);
  result  = replace_all(std::move(result), "%SMEM_CONFIG%", smem_cpp);
  result  = replace_all(std::move(result), "%KEY_BITS%", std::to_string(key_bits));
  result  = replace_all(std::move(result), "%BIT_CONTROL_PARAMETERS%", bit_control_parameters);
  result  = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  *source = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
