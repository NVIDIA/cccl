// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/radix_sort.h"

#include <utility>
#include <vector>

#include "provider/source_common.h"

namespace cuda::coop::tile::detail::provider
{
namespace
{
constexpr int max_radix_sort_bits = 8;

bool fail(std::string message, std::string* error)
{
  if (error)
  {
    *error = std::move(message);
  }
  return false;
}

constexpr char radix_sort_keys_template[] = R"CUDA(
%INCLUDES%

namespace {
#ifndef CUDA_COOP_TILE_BLOCK_THREADS
#define CUDA_COOP_TILE_BLOCK_THREADS 128
#endif
constexpr int kBlockThreads = CUDA_COOP_TILE_BLOCK_THREADS;
#ifndef CUDA_COOP_TILE_ITEMS_PER_THREAD
#define CUDA_COOP_TILE_ITEMS_PER_THREAD 1
#endif
constexpr int kItemsPerThread = CUDA_COOP_TILE_ITEMS_PER_THREAD;
using KeyT = %KEY_T%;
constexpr int kRadixBits = %RADIX_BITS%;
constexpr bool kMemoizeOuterScan = %MEMOIZE_OUTER_SCAN%;
constexpr cub::BlockScanAlgorithm kInnerScanAlgorithm = %INNER_SCAN_ALGO%;
constexpr cudaSharedMemConfig kSmemConfig = %SMEM_CONFIG%;
using BlockRadixSortT = cub::BlockRadixSort<
    KeyT,
    kBlockThreads,
    kItemsPerThread,
    cub::NullType,
    kRadixBits,
    kMemoizeOuterScan,
    kInnerScanAlgorithm,
    kSmemConfig>;
struct ScratchStorage {
    typename BlockRadixSortT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockRadixSort scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockRadixSort alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    KeyT* input_rmem,
%BIT_CONTROL_PARAMETERS%
    KeyT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    if (runtime_begin_bit < 0 || runtime_begin_bit > runtime_end_bit ||
        runtime_end_bit > %KEY_BITS%) {
        asm volatile("trap;");
    }

    KeyT thread_keys[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_keys[i] = input_rmem[i];
    }

%SORT_CALL%
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = thread_keys[i];
    }
    __syncthreads();
}
)CUDA";

constexpr char radix_sort_pairs_template[] = R"CUDA(
%INCLUDES%

namespace {
#ifndef CUDA_COOP_TILE_BLOCK_THREADS
#define CUDA_COOP_TILE_BLOCK_THREADS 128
#endif
constexpr int kBlockThreads = CUDA_COOP_TILE_BLOCK_THREADS;
#ifndef CUDA_COOP_TILE_ITEMS_PER_THREAD
#define CUDA_COOP_TILE_ITEMS_PER_THREAD 1
#endif
constexpr int kItemsPerThread = CUDA_COOP_TILE_ITEMS_PER_THREAD;
using KeyT = %KEY_T%;
using ValueT = %VALUE_T%;
constexpr int kRadixBits = %RADIX_BITS%;
constexpr bool kMemoizeOuterScan = %MEMOIZE_OUTER_SCAN%;
constexpr cub::BlockScanAlgorithm kInnerScanAlgorithm = %INNER_SCAN_ALGO%;
constexpr cudaSharedMemConfig kSmemConfig = %SMEM_CONFIG%;
using BlockRadixSortT = cub::BlockRadixSort<
    KeyT,
    kBlockThreads,
    kItemsPerThread,
    ValueT,
    kRadixBits,
    kMemoizeOuterScan,
    kInnerScanAlgorithm,
    kSmemConfig>;
struct ScratchStorage {
    typename BlockRadixSortT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockRadixSort scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockRadixSort alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    KeyT* input_keys_rmem,
    ValueT* input_vals_rmem,
%BIT_CONTROL_PARAMETERS%
    KeyT* output_keys_rmem,
    ValueT* output_vals_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    if (runtime_begin_bit < 0 || runtime_begin_bit > runtime_end_bit ||
        runtime_end_bit > %KEY_BITS%) {
        asm volatile("trap;");
    }

    KeyT thread_keys[kItemsPerThread];
    ValueT thread_vals[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_keys[i] = input_keys_rmem[i];
        thread_vals[i] = input_vals_rmem[i];
    }

%SORT_CALL%
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_keys_rmem[i] = thread_keys[i];
        output_vals_rmem[i] = thread_vals[i];
    }
    __syncthreads();
}
)CUDA";
} // namespace

// Emit the key-only or paired sort with constant CUB tuning parameters.
// Direction selects the Sort overload; begin/end bit bounds remain runtime
// arguments. Key and value dtype tokens are decoded independently so the
// returned value permutation preserves its original element type.
bool emit_radix_sort_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null RadixSort source destination", error);
  }
  const auto* const spec = std::get_if<RadixSortSpec>(&request.primitive());
  if (!spec)
  {
    return fail("RadixSort emitter received a non-RadixSort PrimitiveSpec", error);
  }

  std::string key_cpp_type;
  std::string key_extra_include;
  std::string unused_key_zero_expression;
  if (!decode_dtype_suffix(spec->key_dtype, &key_cpp_type, &key_extra_include, &unused_key_zero_expression))
  {
    return fail("unsupported RadixSort key dtype=" + spec->key_dtype, error);
  }

  std::string value_cpp_type = "cub::NullType";
  std::string value_extra_include;
  if (spec->pairs)
  {
    std::string unused_value_zero_expression;
    if (!decode_dtype_suffix(spec->value_dtype, &value_cpp_type, &value_extra_include, &unused_value_zero_expression))
    {
      return fail("unsupported RadixSort value dtype=" + spec->value_dtype, error);
    }
  }

  const int key_bits = bitwidth_for_dtype_suffix(spec->key_dtype);
  if (key_bits <= 0)
  {
    return fail("unsupported RadixSort key bit width for dtype=" + spec->key_dtype, error);
  }

  const std::string inner_scan_algorithm = resolve_scan_algorithm_token(spec->inner_scan_algorithm);
  if (inner_scan_algorithm.empty())
  {
    return fail("unsupported RadixSort inner_scan_algorithm=" + spec->inner_scan_algorithm, error);
  }
  const std::string smem_config = resolve_smem_config_token(spec->smem_config);
  if (smem_config.empty())
  {
    return fail("unsupported RadixSort smem_config=" + spec->smem_config, error);
  }
  if (spec->radix_bits <= 0 || spec->radix_bits > max_radix_sort_bits)
  {
    return fail("invalid RadixSort radix_bits=" + std::to_string(spec->radix_bits), error);
  }

  std::string sort_call;
  if (spec->descending)
  {
    sort_call = spec->pairs
                ? "    BlockRadixSortT(scratch.temp_storage).SortDescending(\n"
                  "        thread_keys, thread_vals, runtime_begin_bit, runtime_end_bit);\n"
                : "    BlockRadixSortT(scratch.temp_storage).SortDescending(\n"
                  "        thread_keys, runtime_begin_bit, runtime_end_bit);\n";
  }
  else
  {
    sort_call = spec->pairs
                ? "    BlockRadixSortT(scratch.temp_storage).Sort(\n"
                  "        thread_keys, thread_vals, runtime_begin_bit, runtime_end_bit);\n"
                : "    BlockRadixSortT(scratch.temp_storage).Sort(\n"
                  "        thread_keys, runtime_begin_bit, runtime_end_bit);\n";
  }

  const std::string bit_control_parameters =
    "    int runtime_begin_bit,\n"
    "    int runtime_end_bit,\n";

  const std::vector<std::string> includes = {
    key_extra_include,
    value_extra_include,
    "cuda_runtime.h",
    "cub/block/block_radix_sort.cuh",
  };
  const std::string& entry_name = requested_entry_name;
  std::string result            = spec->pairs ? radix_sort_pairs_template : radix_sort_keys_template;
  result                        = replace_all(std::move(result), "%INCLUDES%", build_include_block(includes));
  result                        = replace_all(std::move(result), "%KEY_T%", key_cpp_type);
  result                        = replace_all(std::move(result), "%VALUE_T%", value_cpp_type);
  result                        = replace_all(std::move(result), "%RADIX_BITS%", std::to_string(spec->radix_bits));
  result  = replace_all(std::move(result), "%MEMOIZE_OUTER_SCAN%", spec->memoize_outer_scan ? "true" : "false");
  result  = replace_all(std::move(result), "%INNER_SCAN_ALGO%", inner_scan_algorithm);
  result  = replace_all(std::move(result), "%SMEM_CONFIG%", smem_config);
  result  = replace_all(std::move(result), "%KEY_BITS%", std::to_string(key_bits));
  result  = replace_all(std::move(result), "%BIT_CONTROL_PARAMETERS%", bit_control_parameters);
  result  = replace_all(std::move(result), "%SORT_CALL%", sort_call);
  result  = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  *source = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
