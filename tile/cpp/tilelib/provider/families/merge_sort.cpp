// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/merge_sort.h"

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

constexpr char merge_sort_keys_template[] = R"CUDA(
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
using ValueT = %VALUE_T%;
using MergeSortT = %MERGE_SORT_TYPE%;
struct CompareOp {
    __device__ __forceinline__ bool operator()(ValueT lhs, ValueT rhs) const {
        return %COMPARE_EXPR%;
    }
};
struct ScratchStorage {
    typename MergeSortT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised MergeSort scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised MergeSort alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    ValueT* input_rmem,
    int runtime_valid_items,
    ValueT runtime_oob_default,
    ValueT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    MergeSortT merge_sort(scratch.temp_storage);

    ValueT thread_keys[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_keys[i] = input_rmem[i];
    }

%SORT_CALL%
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = thread_keys[i];
    }
    %FINAL_SYNC%
}
)CUDA";

constexpr char merge_sort_pairs_template[] = R"CUDA(
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
using MergeSortT = %MERGE_SORT_TYPE%;
struct CompareOp {
    __device__ __forceinline__ bool operator()(KeyT lhs, KeyT rhs) const {
        return %COMPARE_EXPR%;
    }
};
struct ScratchStorage {
    typename MergeSortT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised MergeSort scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised MergeSort alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    KeyT* input_keys_rmem,
    ValueT* input_vals_rmem,
    int runtime_valid_items,
    KeyT runtime_oob_default,
    KeyT* output_keys_rmem,
    ValueT* output_vals_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    MergeSortT merge_sort(scratch.temp_storage);

    KeyT thread_keys[kItemsPerThread];
    ValueT thread_vals[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_keys[i] = input_keys_rmem[i];
        thread_vals[i] = input_vals_rmem[i];
    }

%OOB_FIXUP%    merge_sort.Sort(thread_keys, thread_vals, CompareOp{});

#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_keys_rmem[i] = thread_keys[i];
        output_vals_rmem[i] = thread_vals[i];
    }
    %FINAL_SYNC%
}
)CUDA";
} // namespace

// Select warp or block Merge Sort and emit the matching participation rules.
// The 32-item, one-item-per-thread case uses WarpMergeSort and warp barriers.
// Guarded variants replace invalid keys with the runtime default before the
// sort; paired variants also initialize the corresponding values. The complete
// output extent is preserved, including the padded tail.
bool emit_merge_sort_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null MergeSort source destination", error);
  }
  const auto* const spec = std::get_if<MergeSortSpec>(&request.primitive());
  if (!spec)
  {
    return fail("MergeSort emitter received a non-MergeSort PrimitiveSpec", error);
  }

  const bool warp_scope              = uses_warp_execution_scope(request);
  const std::string include          = warp_scope ? "cub/warp/warp_merge_sort.cuh" : "cub/block/block_merge_sort.cuh";
  const std::string final_sync       = warp_scope ? "__syncwarp();" : "__syncthreads();";
  const std::string participant_rank = warp_scope ? "merge_sort.get_linear_tid()" : "static_cast<int>(threadIdx.x)";

  std::string key_cpp_type;
  std::string key_extra_include;
  std::string unused_key_zero_expression;
  if (!decode_dtype_suffix(spec->key_dtype, &key_cpp_type, &key_extra_include, &unused_key_zero_expression))
  {
    return fail("unsupported MergeSort key dtype=" + spec->key_dtype, error);
  }

  if (!spec->pairs)
  {
    std::string sort_call = "    merge_sort.Sort(thread_keys, CompareOp{});\n";
    if (spec->partial_tile)
    {
      sort_call =
        "    const int max_valid_items = kBlockThreads * kItemsPerThread;\n"
        "    if (runtime_valid_items < 0 || runtime_valid_items > max_valid_items) {\n"
        "        asm volatile(\"trap;\");\n"
        "    }\n"
        "    const int base = %PARTICIPANT_RANK% * kItemsPerThread;\n"
        "#pragma unroll\n"
        "    for (int i = 0; i < kItemsPerThread; ++i) {\n"
        "        if (base + i >= runtime_valid_items)\n"
        "        {\n"
        "            thread_keys[i] = runtime_oob_default;\n"
        "        }\n"
        "    }\n"
        "    merge_sort.Sort(thread_keys, CompareOp{});\n";
    }

    const std::vector<std::string> includes = {key_extra_include, include};
    const std::string& entry_name           = requested_entry_name;
    std::string result                      = merge_sort_keys_template;
    result = replace_all(std::move(result), "%INCLUDES%", build_include_block(includes));
    result = replace_all(std::move(result), "%VALUE_T%", key_cpp_type);
    result = replace_all(
      std::move(result),
      "%MERGE_SORT_TYPE%",
      warp_scope ? "cub::WarpMergeSort<ValueT, kItemsPerThread, kBlockThreads>"
                 : "cub::BlockMergeSort<ValueT, kBlockThreads, kItemsPerThread>");
    result  = replace_all(std::move(result), "%COMPARE_EXPR%", spec->descending ? "lhs > rhs" : "lhs < rhs");
    result  = replace_all(std::move(result), "%SORT_CALL%", sort_call);
    result  = replace_all(std::move(result), "%PARTICIPANT_RANK%", participant_rank);
    result  = replace_all(std::move(result), "%FINAL_SYNC%", final_sync);
    result  = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
    *source = std::move(result);
    return true;
  }

  std::string value_cpp_type;
  std::string value_extra_include;
  std::string value_zero_expression;
  if (!decode_dtype_suffix(spec->value_dtype, &value_cpp_type, &value_extra_include, &value_zero_expression))
  {
    return fail("unsupported MergeSort value dtype=" + spec->value_dtype, error);
  }

  std::string oob_fixup;
  if (spec->partial_tile)
  {
    oob_fixup =
      "    const int max_valid_items = kBlockThreads * kItemsPerThread;\n"
      "    if (runtime_valid_items < 0 || runtime_valid_items > max_valid_items) {\n"
      "        asm volatile(\"trap;\");\n"
      "    }\n"
      "    const int base = %PARTICIPANT_RANK% * kItemsPerThread;\n"
      "#pragma unroll\n"
      "    for (int i = 0; i < kItemsPerThread; ++i) {\n"
      "        const int idx = base + i;\n"
      "        if (idx >= runtime_valid_items)\n"
      "        {\n"
      "            thread_keys[i] = runtime_oob_default;\n"
      "            thread_vals[i] = "
      + value_zero_expression
      + ";\n"
        "        }\n"
        "    }\n\n";
  }

  const std::vector<std::string> includes = {
    key_extra_include,
    value_extra_include,
    include,
  };
  const std::string& entry_name = requested_entry_name;
  std::string result            = merge_sort_pairs_template;
  result                        = replace_all(std::move(result), "%INCLUDES%", build_include_block(includes));
  result                        = replace_all(std::move(result), "%KEY_T%", key_cpp_type);
  result                        = replace_all(std::move(result), "%VALUE_T%", value_cpp_type);
  result                        = replace_all(
    std::move(result),
    "%MERGE_SORT_TYPE%",
    warp_scope ? "cub::WarpMergeSort<KeyT, kItemsPerThread, kBlockThreads, ValueT>"
               : "cub::BlockMergeSort<KeyT, kBlockThreads, kItemsPerThread, ValueT>");
  result  = replace_all(std::move(result), "%OOB_FIXUP%", oob_fixup);
  result  = replace_all(std::move(result), "%PARTICIPANT_RANK%", participant_rank);
  result  = replace_all(std::move(result), "%COMPARE_EXPR%", spec->descending ? "lhs > rhs" : "lhs < rhs");
  result  = replace_all(std::move(result), "%FINAL_SYNC%", final_sync);
  result  = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  *source = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
