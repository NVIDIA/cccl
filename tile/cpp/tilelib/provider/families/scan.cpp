// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/families/scan.h"

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

constexpr char scan_sum_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_scan.cuh>
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
constexpr cub::BlockScanAlgorithm kAlgorithm = %ALGORITHM%;
using BlockScanT = cub::BlockScan<ValueT, kBlockThreads, kAlgorithm>;
%PREFIX_DECL%
struct ScratchStorage {
    typename BlockScanT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockScan scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockScan alignment is too small");
using InputArgT = ValueT*;
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    InputArgT input_rmem,
%RUNTIME_CONTROL_PARAMS%
    ValueT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
%RUNTIME_CONTROL_SETUP%
%PREFIX_SETUP%
    ValueT thread_data[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_data[i] = input_rmem[i];
    }
%SCAN_CALL%
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = thread_data[i];
    }
    __syncthreads();
}
)CUDA";

constexpr char scan_generic_template[] = R"CUDA(
%EXTRA_INCLUDE%
#include <cub/block/block_scan.cuh>
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
constexpr cub::BlockScanAlgorithm kAlgorithm = %ALGORITHM%;
using BlockScanT = cub::BlockScan<ValueT, kBlockThreads, kAlgorithm>;
struct BinaryOpT {
    __device__ __forceinline__ ValueT operator()(ValueT a, ValueT b) const {
        return %BINARY_EXPR%;
    }
};
%PREFIX_DECL%
struct ScratchStorage {
    typename BlockScanT::TempStorage temp_storage;
};
static_assert(sizeof(ScratchStorage) <= %ADVERTISED_SCRATCH_BYTES%,
              "[LP-SCRATCH-001] advertised BlockScan scratch is too small");
static_assert(alignof(ScratchStorage) <= %ADVERTISED_SCRATCH_ALIGNMENT%,
              "[LP-SCRATCH-001] advertised BlockScan alignment is too small");
using InputArgT = ValueT*;
} // namespace

extern "C" __device__ void %ENTRY_NAME%(
    InputArgT input_rmem,
%RUNTIME_CONTROL_PARAMS%
    ValueT* output_rmem,
    void* scratch_smem)
{
    auto& scratch = *reinterpret_cast<ScratchStorage*>(scratch_smem);
%RUNTIME_CONTROL_SETUP%
%PREFIX_SETUP%
    ValueT thread_data[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        thread_data[i] = input_rmem[i];
    }
%SCAN_CALL%
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
        output_rmem[i] = thread_data[i];
    }
    __syncthreads();
}
)CUDA";

std::string sum_prefix_declaration()
{
  return R"CUDA(
struct BlockPrefixCallbackOp {
    ValueT running_total;
    __device__ __forceinline__ explicit BlockPrefixCallbackOp(ValueT value)
        : running_total(value) {}
    __device__ __forceinline__ ValueT operator()(ValueT block_aggregate) {
        ValueT old = running_total;
        running_total = running_total + block_aggregate;
        return old;
    }
};
)CUDA";
}

std::string generic_prefix_declaration()
{
  return R"CUDA(
struct BlockPrefixCallbackOp {
    ValueT running_total;
    BinaryOpT binary_op;
    __device__ __forceinline__ explicit BlockPrefixCallbackOp(ValueT value)
        : running_total(value), binary_op() {}
    __device__ __forceinline__ ValueT operator()(ValueT block_aggregate) {
        ValueT old = running_total;
        running_total = binary_op(running_total, block_aggregate);
        return old;
    }
};
)CUDA";
}

bool make_prefix_fragments(
  const ScanSpec& spec,
  bool generic,
  std::string* declaration,
  std::string* setup,
  std::string* call,
  std::string* error)
{
  if (!declaration || !setup || !call)
  {
    return fail("null Scan prefix destination", error);
  }
  if (!spec.prefix_present)
  {
    declaration->clear();
    setup->clear();
    call->clear();
    return true;
  }

  *declaration = generic ? generic_prefix_declaration() : sum_prefix_declaration();
  *setup       = "    BlockPrefixCallbackOp prefix_op(runtime_prefix);\n";
  *call        = ", prefix_op";
  return true;
}
} // namespace

// Choose the CUB Scan overload from mode, operator, and seed-presence flags.
// Initial and prefix values stay runtime parameters. A prefix uses a per-call
// callback object, so no callback state persists between foreign calls. The
// emitted trailing barrier protects reuse of the compiler-supplied scratch.
bool emit_scan_source(
  const ImplementationSpec& request, const std::string& requested_entry_name, std::string* source, std::string* error)
{
  if (!source)
  {
    return fail("null Scan source destination", error);
  }
  const auto* const spec = std::get_if<ScanSpec>(&request.primitive());
  if (!spec)
  {
    return fail("Scan emitter received a non-Scan PrimitiveSpec", error);
  }
  if (spec->initial_present && spec->prefix_present)
  {
    return fail("Scan cannot set both initial_value and prefix_value", error);
  }

  const std::string algorithm = resolve_scan_algorithm_token(spec->algorithm);
  if (algorithm.empty())
  {
    return fail("unsupported Scan algorithm=" + spec->algorithm, error);
  }

  std::string cpp_type;
  std::string extra_include;
  std::string unused_zero_expression;
  if (!decode_dtype_suffix(spec->dtype, &cpp_type, &extra_include, &unused_zero_expression))
  {
    return fail("unsupported Scan dtype=" + spec->dtype, error);
  }

  std::string prefix_declaration;
  std::string prefix_setup;
  std::string prefix_call;
  if (!make_prefix_fragments(*spec, spec->generic, &prefix_declaration, &prefix_setup, &prefix_call, error))
  {
    return false;
  }

  const std::string runtime_control_parameters = "    ValueT runtime_initial,\n    ValueT runtime_prefix,\n";
  const std::string runtime_control_setup      = "    (void)runtime_initial;\n    (void)runtime_prefix;\n";
  const std::string include_line               = extra_include.empty() ? "" : "#include <" + extra_include + ">\n";
  std::string scan_call;
  std::string binary_expression;
  const char* source_template = scan_sum_template;

  if (!spec->generic)
  {
    if (spec->mode == "exclusive")
    {
      scan_call = "    BlockScanT(scratch.temp_storage).ExclusiveSum(thread_data, thread_data" + prefix_call + ");\n";
    }
    else if (spec->mode == "inclusive")
    {
      scan_call = "    BlockScanT(scratch.temp_storage).InclusiveSum(thread_data, thread_data" + prefix_call + ");\n";
    }
    else
    {
      return fail("unsupported Scan mode=" + spec->mode, error);
    }
  }
  else
  {
    if (!resolve_binary_op_token(spec->scan_op, &binary_expression))
    {
      return fail("unsupported Scan scan_op=" + spec->scan_op, error);
    }

    source_template = scan_generic_template;
    if (spec->mode == "exclusive")
    {
      if (spec->prefix_present)
      {
        scan_call = "    BlockScanT(scratch.temp_storage).ExclusiveScan(thread_data, thread_data, "
                    "BinaryOpT()"
                  + prefix_call + ");\n";
      }
      else
      {
        if (!spec->initial_present)
        {
          return fail("exclusive generic Scan requires an initial or prefix operand", error);
        }
        const std::string initial_clause = ", runtime_initial, BinaryOpT()";
        scan_call = "    BlockScanT(scratch.temp_storage).ExclusiveScan(thread_data, thread_data" + initial_clause
                  + ");\n";
      }
    }
    else if (spec->mode == "inclusive")
    {
      if (spec->initial_present)
      {
        return fail("inclusive generic Scan does not support an initial operand", error);
      }
      scan_call = "    BlockScanT(scratch.temp_storage).InclusiveScan(thread_data, thread_data, BinaryOpT()"
                + prefix_call + ");\n";
    }
    else
    {
      return fail("unsupported Scan mode=" + spec->mode, error);
    }
  }

  const std::string& entry_name = requested_entry_name;
  std::string result            = source_template;
  result                        = replace_all(std::move(result), "%EXTRA_INCLUDE%", include_line);
  result                        = replace_all(std::move(result), "%VALUE_T%", cpp_type);
  result                        = replace_all(std::move(result), "%ALGORITHM%", algorithm);
  result                        = replace_all(std::move(result), "%BINARY_EXPR%", binary_expression);
  result                        = replace_all(std::move(result), "%PREFIX_DECL%", prefix_declaration);
  result                        = replace_all(std::move(result), "%PREFIX_SETUP%", prefix_setup);
  result                        = replace_all(std::move(result), "%SCAN_CALL%", scan_call);
  result  = replace_all(std::move(result), "%RUNTIME_CONTROL_PARAMS%", runtime_control_parameters);
  result  = replace_all(std::move(result), "%RUNTIME_CONTROL_SETUP%", runtime_control_setup);
  result  = replace_all(std::move(result), "%ENTRY_NAME%", entry_name);
  *source = std::move(result);
  return true;
}
} // namespace cuda::coop::tile::detail::provider
