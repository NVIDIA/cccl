// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/shuffle.h"

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

constexpr char shuffle_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_shuffle.cuh>

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
using BlockShuffleT = cub::BlockShuffle<ValueT, kBlockThreads>;
struct ScratchStorage {
    typename BlockShuffleT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockShuffle scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockShuffle alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    const ValueT* input_rmem,
%DISTANCE_PARAMETER%
    ValueT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    (void)distance;
    ValueT thread_data[kItemsPerThread];
    ValueT thread_out[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_data[i] = input_rmem[i];
        thread_out[i] = thread_data[i];
    }
%SHUFFLE_CALL%
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = thread_out[i];
    }
    __syncthreads();
}
)CUDA";
} // namespace

// Initialize outputs from inputs so positions without a neighbor retain their
// value. Offset and rotate require one item per worker, and their distances
// remain runtime operands with generated range checks. Up and down use the
// same ABI slot for distance but ignore it.
bool emit_shuffle_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null Shuffle source destination", error);
  }
  const auto* const spec = std::get_if<ShuffleSpec>(&request.primitive());
  if (!spec)
  {
    return fail("Shuffle emitter received a non-Shuffle PrimitiveSpec", error);
  }
  std::string cpp_type;
  std::string extra_include;
  std::string unused_zero_expression;
  if (!decode_dtype_suffix(spec->dtype, &cpp_type, &extra_include, &unused_zero_expression))
  {
    return fail("unsupported Shuffle dtype=" + spec->dtype, error);
  }

  std::string shuffle_call;
  if (spec->mode == "up")
  {
    shuffle_call = "    BlockShuffleT(scratch.temp_storage).Up(thread_data, thread_out);\n";
  }
  else if (spec->mode == "down")
  {
    shuffle_call = "    BlockShuffleT(scratch.temp_storage).Down(thread_data, thread_out);\n";
  }
  else if (spec->mode == "offset")
  {
    if (request.topology().items_per_thread != 1)
    {
      return fail("Shuffle offset requires one item per thread", error);
    }
    shuffle_call =
      "    if (distance <= -kBlockThreads || distance >= kBlockThreads) {\n"
      "        asm volatile(\"trap;\");\n"
      "    }\n"
      "    BlockShuffleT(scratch.temp_storage).Offset(thread_data[0], thread_out[0], distance);\n";
  }
  else if (spec->mode == "rotate")
  {
    if (request.topology().items_per_thread != 1)
    {
      return fail("Shuffle rotate requires one item per thread", error);
    }
    shuffle_call =
      "    if (distance < 0 || distance >= kBlockThreads) {\n"
      "        asm volatile(\"trap;\");\n"
      "    }\n"
      "    BlockShuffleT(scratch.temp_storage).Rotate(\n"
      "        thread_data[0], thread_out[0], static_cast<unsigned int>(distance));\n";
  }
  else
  {
    return fail("unsupported Shuffle mode=" + spec->mode, error);
  }

  const std::string& entry_name = requested_entry_name;
  std::string result            = shuffle_template;
  result                        = replace_all(
    std::move(result), "%EXTRA_INCLUDE%", extra_include.empty() ? "" : ("#include <" + extra_include + ">\n"));
  result  = replace_all(std::move(result), "%VALUE_T%", cpp_type);
  result  = replace_all(std::move(result), "%DISTANCE_PARAMETER%", "    int distance,\n");
  result  = replace_all(std::move(result), "%SHUFFLE_CALL%", shuffle_call);
  result  = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  *source = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
