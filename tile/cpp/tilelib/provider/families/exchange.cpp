// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/exchange.h"

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

constexpr char exchange_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_exchange.cuh>

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
using BlockExchangeT = cub::BlockExchange<ValueT, kBlockThreads, kItemsPerThread>;
struct ScratchStorage {
    typename BlockExchangeT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockExchange scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockExchange alignment is too small");
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
%PARAMETERS%
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
    ValueT thread_data[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_data[i] = %INPUT_NAME%[i];
    }
    BlockExchangeT(scratch.temp_storage).%METHOD%(thread_data);
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        %OUTPUT_NAME%[i] = thread_data[i];
    }
    __syncthreads();
}
)CUDA";
} // namespace

bool emit_exchange_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null Exchange source destination", error);
  }
  const auto* const spec = std::get_if<ExchangeSpec>(&request.primitive());
  if (!spec)
  {
    return fail("Exchange emitter received a non-Exchange PrimitiveSpec", error);
  }
  std::string cpp_type;
  std::string extra_include;
  std::string unused_zero_expression;
  if (!decode_dtype_suffix(spec->dtype, &cpp_type, &extra_include, &unused_zero_expression))
  {
    return fail("unsupported Exchange dtype=" + spec->dtype, error);
  }

  std::string method;
  if (spec->mode == "striped_to_blocked")
  {
    method = "StripedToBlocked";
  }
  else if (spec->mode == "blocked_to_striped")
  {
    method = "BlockedToStriped";
  }
  else
  {
    return fail("unsupported Exchange mode=" + spec->mode, error);
  }

  const std::string& entry_name = requested_entry_name;
  const std::string parameters  = "    ValueT* input_output_rmem,\n";
  const std::string input_name  = "input_output_rmem";
  const std::string output_name = "input_output_rmem";
  std::string result            = exchange_template;
  result                        = replace_all(
    std::move(result), "%EXTRA_INCLUDE%", extra_include.empty() ? "" : ("#include <" + extra_include + ">\n"));
  result  = replace_all(std::move(result), "%VALUE_T%", cpp_type);
  result  = replace_all(std::move(result), "%METHOD%", method);
  result  = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  result  = replace_all(std::move(result), "%PARAMETERS%", parameters);
  result  = replace_all(std::move(result), "%INPUT_NAME%", input_name);
  result  = replace_all(std::move(result), "%OUTPUT_NAME%", output_name);
  *source = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
