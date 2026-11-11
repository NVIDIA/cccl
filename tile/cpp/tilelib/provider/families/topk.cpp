// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/topk.h"

#include <cstdint>
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

constexpr char topk_keys_template[] = R"CUDA(
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
using BlockTopKT = cub::detail::block_topk<
    KeyT,
    kBlockThreads,
    kItemsPerThread,
    cub::NullType>;
struct ScratchStorage {
    typename BlockTopKT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockTopK scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockTopK alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    KeyT* input_rmem,
    int runtime_k,
    int runtime_num_valid,
    int runtime_begin_bit,
    int runtime_end_bit,
    KeyT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    if (%NUM_VALID_CHECK% ||
        runtime_begin_bit < 0 || runtime_begin_bit >= %KEY_BITS% ||
        runtime_end_bit <= runtime_begin_bit || runtime_end_bit > %KEY_BITS% ||
        runtime_k < 0 || runtime_k > %MAX_ITEMS%) {
        asm volatile("trap;");
        return;
    }

    KeyT thread_keys[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_keys[i] = input_rmem[i];
    }

    const int selected_items = runtime_k < runtime_num_valid ? runtime_k : runtime_num_valid;
    if (selected_items > 0) {
%TOPK_CALL%
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = thread_keys[i];
    }
    __syncthreads();
}
)CUDA";

constexpr char topk_pairs_template[] = R"CUDA(
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
using BlockTopKT = cub::detail::block_topk<
    KeyT,
    kBlockThreads,
    kItemsPerThread,
    ValueT>;
struct ScratchStorage {
    typename BlockTopKT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockTopK scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockTopK alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    KeyT* input_keys_rmem,
    ValueT* input_vals_rmem,
    int runtime_k,
    int runtime_num_valid,
    int runtime_begin_bit,
    int runtime_end_bit,
    KeyT* output_keys_rmem,
    ValueT* output_vals_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    if (%NUM_VALID_CHECK% ||
        runtime_begin_bit < 0 || runtime_begin_bit >= %KEY_BITS% ||
        runtime_end_bit <= runtime_begin_bit || runtime_end_bit > %KEY_BITS% ||
        runtime_k < 0 || runtime_k > %MAX_ITEMS%) {
        asm volatile("trap;");
        return;
    }

    KeyT thread_keys[kItemsPerThread];
    ValueT thread_vals[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_keys[i] = input_keys_rmem[i];
        thread_vals[i] = input_vals_rmem[i];
    }

    const int selected_items = runtime_k < runtime_num_valid ? runtime_k : runtime_num_valid;
    if (selected_items > 0) {
%TOPK_CALL%
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_keys_rmem[i] = thread_keys[i];
        output_vals_rmem[i] = thread_vals[i];
    }
    __syncthreads();
}
)CUDA";
} // namespace

// Select the full-tile or guarded CUB TopK overload at source generation.
// The generated code checks runtime k, valid count, and bit bounds before CUB.
// It writes the full output fragments; only the first min(k, valid_items) logical positions are
// the selected result, and their order is not established by this wrapper.
bool emit_topk_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null TopK source destination", error);
  }
  const auto* const spec = std::get_if<TopKSpec>(&request.primitive());
  if (!spec)
  {
    return fail("TopK emitter received a non-TopK PrimitiveSpec", error);
  }

  std::string key_cpp_type;
  std::string key_extra_include;
  std::string unused_key_zero_expression;
  if (!decode_dtype_suffix(spec->key_dtype, &key_cpp_type, &key_extra_include, &unused_key_zero_expression))
  {
    return fail("unsupported TopK key dtype=" + spec->key_dtype, error);
  }

  std::string value_cpp_type = "cub::NullType";
  std::string value_extra_include;
  if (spec->pairs)
  {
    std::string unused_value_zero_expression;
    if (!decode_dtype_suffix(spec->value_dtype, &value_cpp_type, &value_extra_include, &unused_value_zero_expression))
    {
      return fail("unsupported TopK value dtype=" + spec->value_dtype, error);
    }
  }

  const int key_bits = bitwidth_for_dtype_suffix(spec->key_dtype);
  if (key_bits <= 0)
  {
    return fail("unsupported TopK key bit width for dtype=" + spec->key_dtype, error);
  }

  const int64_t max_items =
    static_cast<int64_t>(request.topology().block_threads) * static_cast<int64_t>(request.topology().items_per_thread);
  if (max_items <= 0)
  {
    return fail("invalid TopK topology", error);
  }

  const std::string num_valid_check =
    spec->full_tile ? "runtime_num_valid != %MAX_ITEMS%" : "runtime_num_valid < 0 || runtime_num_valid > %MAX_ITEMS%";
  const std::string method = std::string(spec->descending ? "max" : "min") + (spec->pairs ? "_pairs" : "_keys");
  std::string topk_call =
    "        BlockTopKT(scratch.temp_storage).template " + method + "<" + (spec->full_tile ? "true" : "false") + ">(";
  if (spec->pairs)
  {
    topk_call += "thread_keys, thread_vals, ";
  }
  else
  {
    topk_call += "thread_keys, ";
  }
  topk_call += "selected_items, runtime_num_valid, runtime_begin_bit, runtime_end_bit);\n";

  const std::vector<std::string> includes = {
    key_extra_include,
    value_extra_include,
    "cub/block/block_topk.cuh",
  };
  const std::string& entry_name = requested_entry_name;
  std::string result            = spec->pairs ? topk_pairs_template : topk_keys_template;
  result                        = replace_all(std::move(result), "%INCLUDES%", build_include_block(includes));
  result                        = replace_all(std::move(result), "%KEY_T%", key_cpp_type);
  result                        = replace_all(std::move(result), "%VALUE_T%", value_cpp_type);
  result                        = replace_all(std::move(result), "%TOPK_CALL%", topk_call);
  result                        = replace_all(std::move(result), "%NUM_VALID_CHECK%", num_valid_check);
  result                        = replace_all(std::move(result), "%MAX_ITEMS%", std::to_string(max_items));
  result                        = replace_all(std::move(result), "%KEY_BITS%", std::to_string(key_bits));
  result                        = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  *source                       = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
