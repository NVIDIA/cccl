// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#include "provider/runtime/runtime.h"
#include "provider/spec_identity.h"

namespace cuda::coop::tile::detail::provider
{
namespace
{
[[nodiscard]] std::string to_string(TileIRStringView value)
{
  if (!value.data || value.len == 0)
  {
    return {};
  }
  return std::string(value.data, value.len);
}

[[nodiscard]] std::string to_string(TileIRDataType dtype)
{
  switch (dtype.kind)
  {
    case TILEIR_DTYPE_I1:
      return "i1";
    case TILEIR_DTYPE_I8:
      return "i8";
    case TILEIR_DTYPE_I16:
      return "i16";
    case TILEIR_DTYPE_I32:
      return "i32";
    case TILEIR_DTYPE_I64:
      return "i64";
    case TILEIR_DTYPE_F16:
      return "f16";
    case TILEIR_DTYPE_BF16:
      return "bf16";
    case TILEIR_DTYPE_F32:
      return "f32";
    case TILEIR_DTYPE_TF32:
      return "tf32";
    case TILEIR_DTYPE_F64:
      return "f64";
    case TILEIR_DTYPE_F8E4M3FN:
      return "f8e4m3fn";
    case TILEIR_DTYPE_F8E5M2:
      return "f8e5m2";
    case TILEIR_DTYPE_F8E8M0FNU:
      return "f8e8m0fnu";
    case TILEIR_DTYPE_F4E2M1FN:
      return "f4e2m1fn";
    case TILEIR_DTYPE_POINTER:
      return "ptr";
    case TILEIR_DTYPE_INVALID:
    default:
      return "";
  }
}
} // namespace

bool trace_enabled()
{
  const char* const value = std::getenv("CUDA_COOP_TILE_PROVIDER_TRACE");
  if (!value)
  {
    return false;
  }
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0;
}

void trace_const_params(const TileIRResolvedCall& call, size_t call_index)
{
  if (!trace_enabled())
  {
    return;
  }
  const std::string impl_identity = call.impl ? to_string(call.impl->symbol) : "";
  const auto* const params        = call.const_arguments;
  const size_t num                = call.num_const_arguments;
  std::fprintf(stderr,
               "[coop_linkage_provider][trace] call=%zu impl=%s const_params=%zu\n",
               call_index,
               impl_identity.c_str(),
               num);
  if (!params && num != 0)
  {
    std::fprintf(stderr, "  [invalid] const_arguments is null with non-zero count\n");
    return;
  }
  for (size_t i = 0; i < num; ++i)
  {
    const auto& param = params[i];
    if (param.kind == TILEIR_CONSTANT_ARGUMENT_STRING)
    {
      std::string preview = to_string(param.string_val);
      if (preview.size() > 80)
      {
        preview = preview.substr(0, 80) + "...";
      }
      std::fprintf(stderr, "  [param %zu] kind=string value=\"%s\"\n", i, preview.c_str());
    }
    else if (param.kind == TILEIR_CONSTANT_ARGUMENT_INT)
    {
      std::fprintf(stderr, "  [param %zu] kind=int value=%lld\n", i, static_cast<long long>(param.int_val));
    }
    else if (param.kind == TILEIR_CONSTANT_ARGUMENT_DTYPE)
    {
      const std::string dtype = to_string(param.dtype_val);
      std::fprintf(stderr, "  [param %zu] kind=dtype value=%s\n", i, dtype.c_str());
    }
    else if (param.kind == TILEIR_CONSTANT_ARGUMENT_FLOAT)
    {
      std::fprintf(stderr, "  [param %zu] kind=float value=%f\n", i, param.float_val);
    }
    else
    {
      std::fprintf(stderr, "  [param %zu] kind=%d\n", i, static_cast<int>(param.kind));
    }
  }
}

// Append one diagnostic line to the configured file and optionally stderr.
// A trace-file setting works independently of the stderr trace flag. File open
// failures are ignored so diagnostics cannot change provider success. Some
// verbose call-parameter traces write directly to stderr and are flag-gated.
void emit_trace_line(const std::string& line)
{
  if (const char* const trace_file = std::getenv("CUDA_COOP_TILE_PROVIDER_TRACE_FILE");
      trace_file && trace_file[0] != '\0')
  {
    if (FILE* const fp = std::fopen(trace_file, "a"))
    {
      std::fprintf(fp, "%s\n", line.c_str());
      std::fclose(fp);
    }
  }
  if (trace_enabled())
  {
    std::fprintf(stderr, "%s\n", line.c_str());
  }
}

void trace_artifact_event(const char* event, const CallMetadata& meta, const std::string& cache_key)
{
  if (!meta.request)
  {
    return;
  }
  const ImplementationSpec& request = *meta.request;
  std::ostringstream oss;
  oss << "[coop_linkage_provider][trace] " << event << " key=" << cache_key
      << " semantic_key=" << semantic_identity(request) << " payload=" << request_payload(request.primitive())
      << " threads=" << request.topology().block_threads << " ipt=" << request.topology().items_per_thread << " sm="
      << request.target().sm_arch.sm_name() << " epoch=" << static_cast<long long>(request.target().cache_epoch);
  emit_trace_line(oss.str());
}

void trace_bundle_event(const char* event, const std::string& bundle_key, size_t primitive_count)
{
  std::ostringstream oss;
  oss << "[coop_linkage_provider][trace] " << event << " key=" << bundle_key << " primitives=" << primitive_count;
  emit_trace_line(oss.str());
}

void trace_resource_event(
  const char* event, const CallMetadata& meta, const PrimitiveResourceEstimate& estimate, uint64_t limit_bytes)
{
  if (!meta.request)
  {
    return;
  }
  const ImplementationSpec& request = *meta.request;
  std::ostringstream oss;
  oss << "[coop_linkage_provider][trace] " << event << " semantic_key=" << semantic_identity(request)
      << " payload=" << request_payload(request.primitive()) << " threads=" << request.topology().block_threads
      << " ipt=" << request.topology().items_per_thread << " sm=" << request.target().sm_arch.sm_name()
      << " temp_storage=" << estimate.temp_storage_bytes << " extra_static_smem=" << estimate.extra_static_smem_bytes
      << " static_smem=" << estimate.static_smem_bytes << " scratch_alignment=" << estimate.scratch_alignment_bytes
      << " limit=" << limit_bytes;
  if (!estimate.detail.empty())
  {
    oss << " detail=" << estimate.detail;
  }
  emit_trace_line(oss.str());
}

void trace_resource_query_batch_event(const char* event, TargetArchitecture target_arch, size_t primitive_count)
{
  std::ostringstream oss;
  oss << "[coop_linkage_provider][trace] " << event << " sm=" << target_arch.sm_name()
      << " primitives=" << primitive_count;
  emit_trace_line(oss.str());
}

void record_nvrtc_compile_counter()
{
  const char* const counter_file = std::getenv("CUDA_COOP_TILE_PROVIDER_NVRTC_COMPILE_COUNTER_FILE");
  if (!counter_file || counter_file[0] == '\0')
  {
    return;
  }
  if (FILE* const fp = std::fopen(counter_file, "a"))
  {
    std::fprintf(fp, "1\n");
    std::fclose(fp);
  }
}
} // namespace cuda::coop::tile::detail::provider
