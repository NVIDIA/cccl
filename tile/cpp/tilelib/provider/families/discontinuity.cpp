// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/discontinuity.h"

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

constexpr char heads_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_discontinuity.cuh>

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
using FlagT = int;
using BlockDiscontinuityT = cub::BlockDiscontinuity<ValueT, kBlockThreads>;
struct InequalityOp {
    __device__ __forceinline__ bool operator()(const ValueT& a, const ValueT& b) const {
        return a != b;
    }
};
struct ScratchStorage {
    typename BlockDiscontinuityT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockDiscontinuity scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockDiscontinuity alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    const ValueT* input_rmem,
    FlagT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    ValueT thread_data[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_data[i] = input_rmem[i];
    }
    FlagT head_flags[kItemsPerThread];
    BlockDiscontinuityT(scratch.temp_storage).FlagHeads(head_flags, thread_data, InequalityOp{});
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = head_flags[i];
    }
    __syncthreads();
}
)CUDA";

constexpr char tails_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_discontinuity.cuh>

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
using FlagT = int;
using BlockDiscontinuityT = cub::BlockDiscontinuity<ValueT, kBlockThreads>;
struct InequalityOp {
    __device__ __forceinline__ bool operator()(const ValueT& a, const ValueT& b) const {
        return a != b;
    }
};
struct ScratchStorage {
    typename BlockDiscontinuityT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockDiscontinuity scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockDiscontinuity alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    const ValueT* input_rmem,
    FlagT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    ValueT thread_data[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_data[i] = input_rmem[i];
    }
    FlagT tail_flags[kItemsPerThread];
    BlockDiscontinuityT(scratch.temp_storage).FlagTails(tail_flags, thread_data, InequalityOp{});
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = tail_flags[i];
    }
    __syncthreads();
}
)CUDA";

constexpr char heads_tails_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_discontinuity.cuh>

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
using FlagT = int;
using BlockDiscontinuityT = cub::BlockDiscontinuity<ValueT, kBlockThreads>;
struct InequalityOp {
    __device__ __forceinline__ bool operator()(const ValueT& a, const ValueT& b) const {
        return a != b;
    }
};
struct ScratchStorage {
    typename BlockDiscontinuityT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockDiscontinuity scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockDiscontinuity alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    const ValueT* input_rmem,
    FlagT* output_heads_rmem,
    FlagT* output_tails_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    ValueT thread_data[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_data[i] = input_rmem[i];
    }
    FlagT head_flags[kItemsPerThread];
    FlagT tail_flags[kItemsPerThread];
    BlockDiscontinuityT(scratch.temp_storage).FlagHeadsAndTails(
        head_flags, tail_flags, thread_data, InequalityOp{});
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_heads_rmem[i] = head_flags[i];
        output_tails_rmem[i] = tail_flags[i];
    }
    __syncthreads();
}
)CUDA";
} // namespace

bool emit_discontinuity_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null Discontinuity source destination", error);
  }
  const auto* const spec = std::get_if<DiscontinuitySpec>(&request.primitive());
  if (!spec)
  {
    return fail("Discontinuity emitter received a non-Discontinuity PrimitiveSpec", error);
  }

  std::string cpp_type;
  std::string extra_include;
  std::string unused_zero_expression;
  if (!decode_dtype_suffix(spec->dtype, &cpp_type, &extra_include, &unused_zero_expression))
  {
    return fail("unsupported Discontinuity dtype=" + spec->dtype, error);
  }

  const char* source_template = nullptr;
  if (spec->mode == "heads")
  {
    source_template = heads_template;
  }
  else if (spec->mode == "tails")
  {
    source_template = tails_template;
  }
  else if (spec->mode == "heads_and_tails")
  {
    source_template = heads_tails_template;
  }
  else
  {
    return fail("unsupported Discontinuity mode=" + spec->mode, error);
  }

  const std::string& entry_name  = requested_entry_name;
  const std::string include_line = extra_include.empty() ? "" : "#include <" + extra_include + ">\n";
  std::string result             = source_template;
  result                         = replace_all(std::move(result), "%EXTRA_INCLUDE%", include_line);
  result                         = replace_all(std::move(result), "%VALUE_T%", cpp_type);
  result                         = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  *source                        = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
