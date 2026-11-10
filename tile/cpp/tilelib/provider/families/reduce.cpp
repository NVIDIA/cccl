// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/reduce.h"

#include <cstdint>
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

constexpr char reduce_sum_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_operators.cuh>

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
constexpr cub::BlockReduceAlgorithm kAlgorithm = %ALGORITHM%;
using BlockReduceT = cub::BlockReduce<ValueT, kBlockThreads, kAlgorithm>;
struct ScratchStorage {
    typename BlockReduceT::TempStorage temp_storage;
    ValueT smem_reduce;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockReduce scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockReduce alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    ValueT* input_rmem,
    ValueT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);

    ValueT thread_sum = %ZERO_EXPR%;
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_sum = thread_sum + input_rmem[i];
    }
    ValueT block_sum = BlockReduceT(scratch.temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0) {
        scratch.smem_reduce = block_sum;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = scratch.smem_reduce;
    }
    __syncthreads();
}
)CUDA";

constexpr char reduce_generic_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_operators.cuh>

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
constexpr cub::BlockReduceAlgorithm kAlgorithm = %ALGORITHM%;
using BlockReduceT = cub::BlockReduce<ValueT, kBlockThreads, kAlgorithm>;

struct BinaryOpT {
    __device__ __forceinline__ ValueT operator()(ValueT a, ValueT b) const {
        return %BINARY_EXPR%;
    }
};
struct ScratchStorage {
    typename BlockReduceT::TempStorage temp_storage;
    ValueT smem_reduce;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockReduce scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockReduce alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    ValueT* input_rmem,
    ValueT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);

    ValueT thread_data[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_data[i] = input_rmem[i];
    }
    ValueT block_reduce = BlockReduceT(scratch.temp_storage).Reduce(thread_data, BinaryOpT());
    if (threadIdx.x == 0) {
        scratch.smem_reduce = block_reduce;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = scratch.smem_reduce;
    }
    __syncthreads();
}
)CUDA";

constexpr char reduce_batched_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/util_ptx.cuh>
#include <cuda/std/span>

namespace {
#ifndef CUDA_COOP_TILE_BLOCK_THREADS
#define CUDA_COOP_TILE_BLOCK_THREADS 32
#endif
constexpr int kBlockThreads = CUDA_COOP_TILE_BLOCK_THREADS;
#ifndef CUDA_COOP_TILE_ITEMS_PER_THREAD
#define CUDA_COOP_TILE_ITEMS_PER_THREAD 1
#endif
constexpr int kItemsPerThread = CUDA_COOP_TILE_ITEMS_PER_THREAD;
constexpr int kValidThreads = %VALID_THREADS%;
using ValueT = %VALUE_T%;
using ReduceT = %REDUCE_TYPE%;
struct BinaryOpT {
    __device__ __forceinline__ ValueT operator()(ValueT a, ValueT b) const {
        return %BINARY_EXPR%;
    }
};
struct ScratchStorage {
    %TEMP_STORAGE%
    %SHARED_RESULT%
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised batched reduction scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised batched reduction alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    ValueT* input_rmem,
    ValueT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    // Start from a real input: synthetic identities would alter some operators'
    // NaN, infinity or signed-zero behavior. Partial lanes are excluded by CUB.
    const ValueT thread_value = cub::ThreadReduce(
        cuda::std::span<ValueT, kItemsPerThread>(input_rmem, kItemsPerThread), BinaryOpT());
    ValueT aggregate = ReduceT(%TEMP_STORAGE_REF%).%REDUCE_CALL%;
    %PUBLISH_RESULT%
}
)CUDA";

constexpr char reduce_warp_batched_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/thread/thread_reduce.cuh>
#include <cub/warp/warp_reduce_batched.cuh>
#include <cub/util_ptx.cuh>
#include <cuda/std/span>

namespace {
#ifndef CUDA_COOP_TILE_BLOCK_THREADS
#define CUDA_COOP_TILE_BLOCK_THREADS 32
#endif
#ifndef CUDA_COOP_TILE_ITEMS_PER_THREAD
#define CUDA_COOP_TILE_ITEMS_PER_THREAD 1
#endif
constexpr int kItemsPerThread = CUDA_COOP_TILE_ITEMS_PER_THREAD;
constexpr int kBatches = %BATCHES_PER_INVOCATION%;
constexpr int kLogicalWarpThreads = %VALID_THREADS%;
using ValueT = %VALUE_T%;
using ReduceT = cub::WarpReduceBatched<ValueT, kBatches, kLogicalWarpThreads>;
struct BinaryOpT {
    __device__ __forceinline__ ValueT operator()(ValueT a, ValueT b) const {
        return %BINARY_EXPR%;
    }
};
struct ScratchStorage {
    typename ReduceT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised batched reduction scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised batched reduction alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    ValueT* input_rmem,
    ValueT* output_rmem,
    void* scratch_smem)
{
    ValueT partials[kBatches];
#pragma unroll
    for (int batch = 0; batch < kBatches; ++batch) {
        partials[batch] = cub::ThreadReduce(
            cuda::std::span<ValueT, kItemsPerThread>(
                input_rmem + batch * kItemsPerThread, kItemsPerThread), BinaryOpT());
    }
    // WarpReduceBatched uses no shared storage. Each logical warp can keep
    // its empty TempStorage locally without a shared-storage reuse barrier.
    typename ReduceT::TempStorage temp_storage;
    const ValueT aggregate = ReduceT(temp_storage).%REDUCE_CALL%;
    // CUB distributes the two aggregates into lanes 0 and 1. Replicate them
    // into the even/odd output lanes described by the fragment map.
    output_rmem[0] = cub::ShuffleIndex<32>(aggregate, threadIdx.x % kBatches, 0xffffffffu);
}
)CUDA";

bool emit_batched_reduce_source(
  const ImplementationSpec& request,
  const ReduceSpec& spec,
  const std::string& requested_entry_name,
  const std::string& include_line,
  const std::string& cpp_type,
  const std::string& algorithm,
  std::string* source,
  std::string* error)
{
  std::string binary_expression = "a + b";
  if (spec.generic && !resolve_binary_op_token(spec.binary_op, &binary_expression))
  {
    return fail("unsupported Reduce binary_op=" + spec.binary_op, error);
  }
  const bool warp          = uses_warp_execution_scope(request);
  const int64_t row_extent = request.demand().extent() / spec.batches;
  const int64_t valid_threads =
    row_extent < request.topology().block_threads ? row_extent : request.topology().block_threads;
  if (request.topology().reduction_batches_per_invocation > 1)
  {
    std::string result = reduce_warp_batched_template;
    result             = replace_all(std::move(result), "%EXTRA_INCLUDE%", include_line);
    result             = replace_all(std::move(result), "%VALUE_T%", cpp_type);
    result             = replace_all(std::move(result), "%VALID_THREADS%", std::to_string(valid_threads));
    result             = replace_all(std::move(result),
                                     "%BATCHES_PER_INVOCATION%",
                                     std::to_string(request.topology().reduction_batches_per_invocation));
    result             = replace_all(std::move(result), "%BINARY_EXPR%", binary_expression);
    result             = replace_all(std::move(result), "%ENTRY_NAME%", requested_entry_name);
    result =
      replace_all(std::move(result), "%REDUCE_CALL%", spec.generic ? "Reduce(partials, BinaryOpT())" : "Sum(partials)");
    *source = std::move(result);
    return true;
  }
  std::string result = reduce_batched_template;
  result             = replace_all(std::move(result), "%EXTRA_INCLUDE%", include_line);
  result             = replace_all(std::move(result), "%VALUE_T%", cpp_type);
  result             = replace_all(std::move(result), "%VALID_THREADS%", std::to_string(valid_threads));
  result             = replace_all(std::move(result), "%BINARY_EXPR%", binary_expression);
  result             = replace_all(std::move(result), "%ENTRY_NAME%", requested_entry_name);
  result             = replace_all(
    std::move(result),
    "%REDUCE_TYPE%",
    warp ? "cub::WarpReduce<ValueT, 32>" : "cub::BlockReduce<ValueT, kBlockThreads, " + algorithm + ">");
  // The compiler gives all invocations the same CTA scratch base. Independent
  // physical warps need independent storage; a warp may reuse its slot only
  // after the final warp barrier. CUDA supports at most 32 warps per CTA.
  result = replace_all(
    std::move(result),
    "%TEMP_STORAGE%",
    warp ? "typename ReduceT::TempStorage temp_storage[32];" : "typename ReduceT::TempStorage temp_storage;");
  result = replace_all(std::move(result), "%SHARED_RESULT%", warp ? "" : "ValueT aggregate;");
  result = replace_all(
    std::move(result), "%TEMP_STORAGE_REF%", warp ? "scratch.temp_storage[threadIdx.x / 32]" : "scratch.temp_storage");
  result = replace_all(
    std::move(result),
    "%REDUCE_CALL%",
    spec.generic ? "Reduce(thread_value, BinaryOpT(), kValidThreads)" : "Sum(thread_value, kValidThreads)");
  result = replace_all(
    std::move(result),
    "%PUBLISH_RESULT%",
    warp ? "output_rmem[0] = cub::ShuffleIndex<32>(aggregate, 0, 0xffffffffu);\n"
           "    __syncwarp();"
         : "if (threadIdx.x == 0) {\n"
           "        scratch.aggregate = aggregate;\n"
           "    }\n"
           "    __syncthreads();\n"
           "    output_rmem[0] = scratch.aggregate;\n"
           "    __syncthreads();");
  *source = std::move(result);
  return true;
}
} // namespace

// Emit native row reductions, pairing rows when WarpReduceBatched is selected.
// Preserve the original CTA reduction and input-shaped broadcast for legacy
// declarations; its shared aggregate and final barrier protect scratch reuse.
bool emit_reduce_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null Reduce source destination", error);
  }
  const auto* const spec = std::get_if<ReduceSpec>(&request.primitive());
  if (!spec)
  {
    return fail("Reduce emitter received a non-Reduce PrimitiveSpec", error);
  }

  const std::string algorithm = resolve_reduce_algorithm_token(spec->algorithm);
  if (algorithm.empty())
  {
    return fail("unsupported Reduce algorithm=" + spec->algorithm, error);
  }

  std::string cpp_type;
  std::string extra_include;
  std::string zero_expression;
  if (!decode_dtype_suffix(spec->dtype, &cpp_type, &extra_include, &zero_expression))
  {
    return fail("unsupported Reduce dtype=" + spec->dtype, error);
  }

  const std::string include_line = extra_include.empty() ? "" : "#include <" + extra_include + ">\n";
  if (spec->batches > 0)
  {
    return emit_batched_reduce_source(
      request, *spec, requested_entry_name, include_line, cpp_type, algorithm, source, error);
  }
  if (!spec->generic)
  {
    const std::string& entry_name = requested_entry_name;
    std::string result            = reduce_sum_template;
    result                        = replace_all(std::move(result), "%EXTRA_INCLUDE%", include_line);
    result                        = replace_all(std::move(result), "%VALUE_T%", cpp_type);
    result                        = replace_all(std::move(result), "%ALGORITHM%", algorithm);
    result                        = replace_all(std::move(result), "%ZERO_EXPR%", zero_expression);
    result                        = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
    *source                       = std::move(result);
    return true;
  }

  std::string binary_expression;
  if (!resolve_binary_op_token(spec->binary_op, &binary_expression))
  {
    return fail("unsupported Reduce binary_op=" + spec->binary_op, error);
  }
  const std::string& entry_name = requested_entry_name;
  std::string result            = reduce_generic_template;
  result                        = replace_all(std::move(result), "%EXTRA_INCLUDE%", include_line);
  result                        = replace_all(std::move(result), "%VALUE_T%", cpp_type);
  result                        = replace_all(std::move(result), "%ALGORITHM%", algorithm);
  result                        = replace_all(std::move(result), "%BINARY_EXPR%", binary_expression);
  result                        = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  *source                       = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
