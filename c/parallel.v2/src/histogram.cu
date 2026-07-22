//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <cccl/c/histogram.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>
#include <util/serialization.h>

using namespace hostjit::codegen;

// JIT wrapper produced by CubCall:
//   fn(temp, temp_bytes,
//      d_samples,                  // input iterator state
//      d_histogram,                // output pointer (counter_t*)
//      &num_levels,                // int (host pointer)
//      &lower_level, &upper_level, // level_t (host pointer)
//      &num_row_pixels,            // long long (host pointer)
//      &num_rows,                  // long long (host pointer)
//      &row_stride_bytes,          // size_t (host-precomputed: row_stride_samples * sizeof(sample_t))
//      stream)
using histogram_fn_t = int (*)(void*, size_t*, void*, void*, void*, void*, void*, void*, void*, void*, void*);

static constexpr cccl_type_info k_int_type{sizeof(int), alignof(int), CCCL_INT32};
static constexpr cccl_type_info k_int64_type{sizeof(long long), alignof(long long), CCCL_INT64};
static constexpr cccl_type_info k_size_type{sizeof(unsigned long long), alignof(unsigned long long), CCCL_UINT64};

CUresult cccl_device_histogram_build_ex(
  cccl_device_histogram_build_result_t* build_ptr,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int /*num_output_levels_val*/,
  cccl_iterator_t d_output_histograms,
  cccl_type_info level_type,
  int64_t /*num_rows*/,
  int64_t /*row_stride_samples*/,
  bool /*is_evenly_segmented*/,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  if (num_channels != 1 || num_active_channels != 1)
  {
    fprintf(stderr,
            "\nERROR in cccl_device_histogram_build(): only num_channels=1, num_active_channels=1 is "
            "supported in the HostJIT path.\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  // level_t comes from the build-time type info. CUB infers
  // sample_t / counter_t from the iterator and output pointer respectively.
  CubCallResult result =
    CubCall::from("cub/device/device_histogram.cuh")
      .run("cub::DeviceHistogram::HistogramEven")
      .name("cccl_jit_histogram_even")
      .with(temp_storage,
            temp_bytes,
            in(d_samples),
            out(d_output_histograms),
            typed_scalar(k_int_type, "num_levels"),
            typed_scalar(level_type, "lower_level"),
            typed_scalar(level_type, "upper_level"),
            typed_scalar(k_int64_type, "num_row_pixels"),
            typed_scalar(k_int64_type, "num_rows"),
            typed_scalar(k_size_type, "row_stride_bytes"),
            stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build_ptr->cc = cc_major * 10 + cc_minor;
  // Populate payload with the compiled artifact bytes too (not just the
  // live loaded state) so cccl_device_histogram_serialize works uniformly
  // whether the build_result came from this fused path or from
  // _compile()+_load(). Best-effort: a failed read leaves payload null,
  // which only affects later serialize() calls, not this build itself.
  auto library_bytes = cccl::detail::read_compiled_library_bytes(result.compiler);
  cccl::detail::copy_bytes(library_bytes, build_ptr->payload, build_ptr->payload_size);
  build_ptr->jit_compiler        = result.compiler;
  build_ptr->histogram_fn        = result.fn_ptr;
  build_ptr->counter_type        = d_output_histograms.value_type;
  build_ptr->level_type          = level_type;
  build_ptr->sample_type         = d_samples.value_type;
  build_ptr->num_channels        = num_channels;
  build_ptr->num_active_channels = num_active_channels;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_histogram_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_histogram_compile(
  cccl_device_histogram_build_result_t* build_ptr,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int /*num_output_levels_val*/,
  cccl_iterator_t d_output_histograms,
  cccl_type_info level_type,
  int64_t /*num_rows*/,
  int64_t /*row_stride_samples*/,
  bool /*is_evenly_segmented*/,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  if (num_channels != 1 || num_active_channels != 1)
  {
    fprintf(stderr,
            "\nERROR in cccl_device_histogram_compile(): only num_channels=1, num_active_channels=1 is "
            "supported in the HostJIT path.\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  // level_t comes from the build-time type info. CUB infers
  // sample_t / counter_t from the iterator and output pointer respectively.
  CubCallCompileOnlyResult result =
    CubCall::from("cub/device/device_histogram.cuh")
      .run("cub::DeviceHistogram::HistogramEven")
      .name("cccl_jit_histogram_even")
      .with(temp_storage,
            temp_bytes,
            in(d_samples),
            out(d_output_histograms),
            typed_scalar(k_int_type, "num_levels"),
            typed_scalar(level_type, "lower_level"),
            typed_scalar(level_type, "upper_level"),
            typed_scalar(k_int64_type, "num_row_pixels"),
            typed_scalar(k_int64_type, "num_rows"),
            typed_scalar(k_size_type, "row_stride_bytes"),
            stream)
      .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build_ptr->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_bytes(result.library_bytes, build_ptr->payload, build_ptr->payload_size);
  // Zero-init fields set by _load, not _compile (matches v1's contract).
  build_ptr->jit_compiler        = nullptr;
  build_ptr->histogram_fn        = nullptr;
  build_ptr->counter_type        = d_output_histograms.value_type;
  build_ptr->level_type          = level_type;
  build_ptr->sample_type         = d_samples.value_type;
  build_ptr->num_channels        = num_channels;
  build_ptr->num_active_channels = num_active_channels;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_histogram_compile(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_histogram_load(cccl_device_histogram_build_result_t* build_ptr, const char* ctk_path)
try
{
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  hostjit::CompilerConfig jit_config = cccl::detail::make_load_jit_config(ctk_path);

  auto compiler = std::make_unique<hostjit::JITCompiler>(jit_config);
  std::vector<char> library_bytes(
    static_cast<char*>(build_ptr->payload), static_cast<char*>(build_ptr->payload) + build_ptr->payload_size);
  if (!compiler->loadFromBytes(library_bytes))
  {
    fprintf(stderr, "\nERROR in cccl_device_histogram_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  auto fn = compiler->getFunction<histogram_fn_t>("cccl_jit_histogram_even");
  if (!fn)
  {
    fprintf(stderr, "\nERROR in cccl_device_histogram_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  build_ptr->jit_compiler = compiler.release();
  build_ptr->histogram_fn = reinterpret_cast<void*>(fn);

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_histogram_load(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_histogram_build(
  cccl_device_histogram_build_result_t* build,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int num_output_levels_val,
  cccl_iterator_t d_output_histograms,
  cccl_type_info level_type,
  int64_t num_rows,
  int64_t row_stride_samples,
  bool is_evenly_segmented,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_histogram_build_ex(
    build,
    num_channels,
    num_active_channels,
    d_samples,
    num_output_levels_val,
    d_output_histograms,
    level_type,
    num_rows,
    row_stride_samples,
    is_evenly_segmented,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_histogram_even(
  cccl_device_histogram_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_samples,
  cccl_iterator_t d_output_histograms,
  cccl_value_t num_output_levels,
  cccl_value_t lower_level,
  cccl_value_t upper_level,
  int64_t num_row_pixels,
  int64_t num_rows,
  int64_t row_stride_samples,
  CUstream stream)
try
{
  if (!build.histogram_fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // CUB takes row_stride_bytes (not samples). Pre-compute on the host so the
  // JIT wrapper doesn't need a sizeof(sample_t) computation.
  long long num_row_pixels_ll = static_cast<long long>(num_row_pixels);
  long long num_rows_ll       = static_cast<long long>(num_rows);
  size_t row_stride_bytes     = static_cast<size_t>(row_stride_samples) * build.sample_type.size;

  auto fn          = reinterpret_cast<histogram_fn_t>(build.histogram_fn);
  const int status = fn(
    d_temp_storage,
    temp_storage_bytes,
    d_samples.state,
    d_output_histograms.state,
    num_output_levels.state,
    lower_level.state,
    upper_level.state,
    &num_row_pixels_ll,
    &num_rows_ll,
    &row_stride_bytes,
    reinterpret_cast<void*>(stream));

  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_histogram_even(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_histogram_cleanup(cccl_device_histogram_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->histogram_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_histogram_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult
cccl_device_histogram_serialize(const cccl_device_histogram_build_result_t* build, void** out_buf, size_t* out_size)
try
{
  *out_buf  = nullptr;
  *out_size = 0;

  if (build == nullptr || build->payload == nullptr || build->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // extra = {counter_type (size,alignment,type: 3xu64/u32), level_type (3x...),
  //          sample_type (3x...), num_channels (i32), num_active_channels (i32)}
  std::vector<char> extra;
  auto append_type = [&](cccl_type_info t) {
    uint64_t size = t.size, alignment = t.alignment;
    uint32_t type = static_cast<uint32_t>(t.type);
    extra.insert(extra.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
    extra.insert(
      extra.end(), reinterpret_cast<char*>(&alignment), reinterpret_cast<char*>(&alignment) + sizeof(alignment));
    extra.insert(extra.end(), reinterpret_cast<char*>(&type), reinterpret_cast<char*>(&type) + sizeof(type));
  };
  append_type(build->counter_type);
  append_type(build->level_type);
  append_type(build->sample_type);
  int32_t num_channels_i32        = build->num_channels;
  int32_t num_active_channels_i32 = build->num_active_channels;
  extra.insert(extra.end(),
               reinterpret_cast<char*>(&num_channels_i32),
               reinterpret_cast<char*>(&num_channels_i32) + sizeof(num_channels_i32));
  extra.insert(extra.end(),
               reinterpret_cast<char*>(&num_active_channels_i32),
               reinterpret_cast<char*>(&num_active_channels_i32) + sizeof(num_active_channels_i32));

  std::vector<char> blob = cccl::serialization_v2::write_blob(
    CCCL_SERIALIZATION_V2_ALGO_HISTOGRAM,
    build->cc,
    {"cccl_jit_histogram_even"},
    extra,
    build->payload,
    build->payload_size);

  auto* buf = new char[blob.size()];
  std::memcpy(buf, blob.data(), blob.size());
  *out_buf  = buf;
  *out_size = blob.size();

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_histogram_serialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_histogram_deserialize(cccl_device_histogram_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::serialization_v2::parsed_blob parsed =
    cccl::serialization_v2::read_blob(CCCL_SERIALIZATION_V2_ALGO_HISTOGRAM, buf, size);

  // Commit-on-success: build a local result{} and only assign to
  // *build_ptr after every read succeeds, so *build_ptr is left unchanged
  // on failure (matches v1's deserialize contract).
  constexpr size_t type_sz  = sizeof(uint64_t) * 2 + sizeof(uint32_t);
  constexpr size_t extra_sz = type_sz * 3 + sizeof(int32_t) * 2;
  if (parsed.extra.size() != extra_sz)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  size_t pos     = 0;
  auto read_type = [&](cccl_type_info& out) {
    uint64_t size, alignment;
    uint32_t type;
    std::memcpy(&size, parsed.extra.data() + pos, sizeof(size));
    pos += sizeof(size);
    std::memcpy(&alignment, parsed.extra.data() + pos, sizeof(alignment));
    pos += sizeof(alignment);
    std::memcpy(&type, parsed.extra.data() + pos, sizeof(type));
    pos += sizeof(type);
    out = cccl_type_info{static_cast<size_t>(size), static_cast<size_t>(alignment), static_cast<cccl_type_enum>(type)};
  };

  cccl_device_histogram_build_result_t result{};
  read_type(result.counter_type);
  read_type(result.level_type);
  read_type(result.sample_type);
  int32_t num_channels_i32, num_active_channels_i32;
  std::memcpy(&num_channels_i32, parsed.extra.data() + pos, sizeof(num_channels_i32));
  pos += sizeof(num_channels_i32);
  std::memcpy(&num_active_channels_i32, parsed.extra.data() + pos, sizeof(num_active_channels_i32));
  pos += sizeof(num_active_channels_i32);

  result.cc = parsed.cc;
  cccl::detail::copy_bytes(
    std::vector<char>(parsed.payload, parsed.payload + parsed.payload_size), result.payload, result.payload_size);
  result.jit_compiler        = nullptr;
  result.histogram_fn        = nullptr;
  result.num_channels        = num_channels_i32;
  result.num_active_channels = num_active_channels_i32;

  *build_ptr = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_histogram_deserialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
