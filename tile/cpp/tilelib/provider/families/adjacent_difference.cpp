// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/adjacent_difference.h"

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

constexpr char adjacent_difference_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <%PRIMITIVE_HEADER%>

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
%PRIMITIVE_TYPE%
struct DifferenceOp {
    __device__ __forceinline__ ValueT operator()(const ValueT& a, const ValueT& b) const {
        return a - b;
    }
};
struct ScratchStorage {
    %TEMP_STORAGE_TYPE% temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockAdjacentDifference scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockAdjacentDifference alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    const ValueT* input_rmem,
    ValueT* output_rmem,
    void* scratch_smem)
{
    %SCRATCH_REFERENCE%
    ValueT thread_data[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_data[i] = input_rmem[i];
    }
    ValueT thread_out[kItemsPerThread];
%DIFFERENCE_CALL%
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = thread_out[i];
    }
    %FINAL_SYNC%
}
)CUDA";

constexpr char warp_subtract_left[] = R"CUDA(
    const int lane = static_cast<int>(threadIdx.x % 32);
    const ValueT predecessor = cub::ShuffleUp<32>(thread_data[kItemsPerThread - 1], 1, 0, 0xffffffffu);
    thread_out[0] = lane == 0 ? thread_data[0] : DifferenceOp{}(thread_data[0], predecessor);
#pragma unroll
    for (int i = 1; i < kItemsPerThread; ++i) {
        thread_out[i] = DifferenceOp{}(thread_data[i], thread_data[i - 1]);
    }
)CUDA";

constexpr char warp_subtract_right[] = R"CUDA(
    const int lane = static_cast<int>(threadIdx.x % 32);
    const ValueT successor = cub::ShuffleDown<32>(thread_data[0], 1, 31, 0xffffffffu);
#pragma unroll
    for (int i = 0; i < kItemsPerThread - 1; ++i) {
        thread_out[i] = DifferenceOp{}(thread_data[i], thread_data[i + 1]);
    }
    thread_out[kItemsPerThread - 1] = lane == 31 ? thread_data[kItemsPerThread - 1]
        : DifferenceOp{}(thread_data[kItemsPerThread - 1], successor);
)CUDA";
} // namespace

bool emit_adjacent_difference_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null AdjacentDifference source destination", error);
  }
  const auto* const spec = std::get_if<AdjacentDifferenceSpec>(&request.primitive());
  if (!spec)
  {
    return fail("AdjacentDifference emitter received a non-AdjacentDifference PrimitiveSpec", error);
  }

  std::string cpp_type;
  std::string extra_include;
  std::string unused_zero_expression;
  if (!decode_dtype_suffix(spec->dtype, &cpp_type, &extra_include, &unused_zero_expression))
  {
    return fail("unsupported AdjacentDifference dtype=" + spec->dtype, error);
  }

  std::string method;
  if (spec->mode == "subtract_left")
  {
    method = "SubtractLeft";
  }
  else if (spec->mode == "subtract_right")
  {
    method = "SubtractRight";
  }
  else
  {
    return fail("unsupported AdjacentDifference mode=" + spec->mode, error);
  }

  const std::string& entry_name  = requested_entry_name;
  const std::string include_line = extra_include.empty() ? "" : "#include <" + extra_include + ">\n";
  const bool warp_scope          = uses_warp_execution_scope(request);
  std::string result             = adjacent_difference_template;
  result                         = replace_all(std::move(result), "%EXTRA_INCLUDE%", include_line);
  result                         = replace_all(std::move(result), "%VALUE_T%", cpp_type);
  result = replace_all(std::move(result),
                       "%PRIMITIVE_HEADER%",
                       warp_scope ? "cub/util_ptx.cuh" : "cub/block/block_adjacent_difference.cuh");
  result = replace_all(
    std::move(result),
    "%PRIMITIVE_TYPE%",
    warp_scope ? "" : "using BlockAdjacentDifferenceT = cub::BlockAdjacentDifference<ValueT, kBlockThreads>;");
  result = replace_all(std::move(result),
                       "%TEMP_STORAGE_TYPE%",
                       warp_scope ? "unsigned char" : "typename BlockAdjacentDifferenceT::TempStorage");
  result = replace_all(std::move(result),
                       "%SCRATCH_REFERENCE%",
                       warp_scope ? "" : "auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);");
  result = replace_all(
    std::move(result),
    "%DIFFERENCE_CALL%",
    warp_scope
      ? (spec->mode == "subtract_left" ? warp_subtract_left : warp_subtract_right)
      : "    BlockAdjacentDifferenceT(scratch.temp_storage)." + method + "(thread_data, thread_out, DifferenceOp{});");
  result  = replace_all(std::move(result), "%FINAL_SYNC%", warp_scope ? "__syncwarp();" : "__syncthreads();");
  result  = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  *source = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
