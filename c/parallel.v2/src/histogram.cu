//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstring>
#include <format>
#include <string>

#include <cccl/c/histogram.h>
#include <hostjit/codegen/bitcode.hpp>
#include <hostjit/codegen/iterators.hpp>
#include <hostjit/codegen/types.hpp>
#include <hostjit/jit_compiler.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

// ---------------------------------------------------------------------------
// JIT source generation
// ---------------------------------------------------------------------------
// The JIT function signature for a single-channel HistogramEven call:
//
//   int cccl_jit_histogram_even(
//       void* d_temp_storage, size_t* temp_storage_bytes,
//       void* d_samples_ptr,        // raw pointer (CCCL_POINTER) or state bytes (CCCL_ITERATOR)
//       void* d_histogram_ptr,      // counter_t*
//       void* num_levels_host_ptr,  // int* (host pointer to num_output_levels)
//       void* lower_level_host_ptr, // level_t* (host pointer)
//       void* upper_level_host_ptr, // level_t* (host pointer)
//       long long num_row_pixels,
//       long long num_rows,
//       long long row_stride_samples,  // stride in units of samples
//       void* stream)
//
// row_stride_bytes = row_stride_samples * sizeof(sample_t) is computed inside.

static const char* k_export_macro = R"(
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
)";

static std::string make_histogram_even_source(
  cccl_iterator_t d_samples,
  const std::string& sample_type,
  const std::string& counter_type,
  const std::string& level_type)
{
  // Generate iterator setup for the samples input (handles pointer and custom iterators).
  auto it_code =
    make_input_iterator(d_samples, sample_type, sample_type, "samples_it_t", "samples_it", "d_samples_ptr");

  return std::format(
    R"SRC(
#include <cuda_runtime.h>
#include <cuda/std/iterator>
#include <cub/device/device_histogram.cuh>
{0}
{1}
extern "C" EXPORT int cccl_jit_histogram_even(
    void* d_temp_storage, size_t* temp_storage_bytes,
    void* d_samples_ptr,
    void* d_histogram_ptr,
    void* num_levels_host_ptr,
    void* lower_level_host_ptr,
    void* upper_level_host_ptr,
    long long num_row_pixels,
    long long num_rows,
    long long row_stride_samples,
    void* stream)
{{
    using sample_t  = {2};
    using counter_t = {3};
    using level_t   = {4};

    {5}

    int num_levels = 0;
    __builtin_memcpy(&num_levels, num_levels_host_ptr, sizeof(int));

    level_t lower_level, upper_level;
    __builtin_memcpy(&lower_level, lower_level_host_ptr, sizeof(level_t));
    __builtin_memcpy(&upper_level, upper_level_host_ptr, sizeof(level_t));

    // row_stride_bytes: stride in bytes (CUB expects bytes, not elements)
    size_t row_stride_bytes = static_cast<size_t>(row_stride_samples) * sizeof(sample_t);

    cudaError_t err = cub::DeviceHistogram::HistogramEven(
        d_temp_storage, *temp_storage_bytes,
        samples_it,
        static_cast<counter_t*>(d_histogram_ptr),
        num_levels, lower_level, upper_level,
        static_cast<long long>(num_row_pixels),
        static_cast<long long>(num_rows),
        row_stride_bytes,
        static_cast<cudaStream_t>(stream));
    return static_cast<int>(err);
}}
)SRC",
    k_export_macro,
    it_code.preamble,
    sample_type,
    counter_type,
    level_type,
    it_code.setup_code);
}

// ---------------------------------------------------------------------------
// Runtime function typedef
// ---------------------------------------------------------------------------

// (temp, bytes, samples, histogram, num_levels_host_ptr, lower_host_ptr, upper_host_ptr,
//  num_row_pixels, num_rows, row_stride_samples, stream)
using histogram_fn_t =
  int (*)(void*, size_t*, void*, void*, void*, void*, void*, long long, long long, long long, void*);

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_histogram_build_ex(
  cccl_device_histogram_build_result_t* build_ptr,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int /*num_output_levels_val*/,
  cccl_iterator_t d_output_histograms,
  cccl_value_t lower_level,
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
            "supported in the ClangJIT path.\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  std::string sample_type = get_type_name(d_samples.value_type.type);
  if (sample_type.empty())
  {
    fprintf(stderr, "\nERROR in cccl_device_histogram_build(): unsupported sample type\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string counter_type = get_type_name(d_output_histograms.value_type.type);
  if (counter_type.empty())
  {
    fprintf(stderr, "\nERROR in cccl_device_histogram_build(): unsupported counter type\n");
    return CUDA_ERROR_UNKNOWN;
  }

  // The level type comes from the lower_level value's type
  std::string level_type = get_type_name(lower_level.type.type);
  if (level_type.empty())
  {
    // Fall back to sample type if level type is unknown
    level_type = sample_type;
  }

  std::string source = make_histogram_even_source(d_samples, sample_type, counter_type, level_type);

  // Build compiler config and link any iterator bitcode (e.g. for ConstantIterator).
  auto jit_config = cccl::detail::make_jit_config(
    cc_major, cc_minor, ctk_root, cccl_include_path, merged.get(), "cccl_jit_histogram_even");
  {
    BitcodeCollector bitcode(jit_config, reinterpret_cast<uintptr_t>(build_ptr));
    bitcode.add_iterator(d_samples, "samples");
    // bitcode files are written to jit_config.device_bitcode_files; cleanup temp files after compile
    auto* compiler = new hostjit::JITCompiler(jit_config);
    if (!compiler->compile(source))
    {
      fprintf(stderr, "\nJIT compilation failed: %s\n", compiler->getLastError().c_str());
      delete compiler;
      bitcode.cleanup();
      return CUDA_ERROR_UNKNOWN;
    }
    bitcode.cleanup();

    void* fn_ptr = compiler->getFunction<void*>("cccl_jit_histogram_even");
    if (!fn_ptr)
    {
      fprintf(
        stderr, "\nJIT symbol lookup failed for 'cccl_jit_histogram_even': %s\n", compiler->getLastError().c_str());
      delete compiler;
      return CUDA_ERROR_UNKNOWN;
    }

    build_ptr->cc                  = cc_major * 10 + cc_minor;
    build_ptr->cubin               = cccl::detail::copy_cubin(compiler->getCubin(), &build_ptr->cubin_size);
    build_ptr->jit_compiler        = compiler;
    build_ptr->histogram_fn        = fn_ptr;
    build_ptr->counter_type        = d_output_histograms.value_type;
    build_ptr->level_type          = lower_level.type;
    build_ptr->sample_type         = d_samples.value_type;
    build_ptr->num_channels        = num_channels;
    build_ptr->num_active_channels = num_active_channels;

    return CUDA_SUCCESS;
  }
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_histogram_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_histogram_build(
  cccl_device_histogram_build_result_t* build,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int num_output_levels_val,
  cccl_iterator_t d_output_histograms,
  cccl_value_t lower_level,
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
    lower_level,
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

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

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
{
  try
  {
    if (!build.histogram_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    auto fn    = reinterpret_cast<histogram_fn_t>(build.histogram_fn);
    int status = fn(
      d_temp_storage,
      temp_storage_bytes,
      d_samples.state,
      d_output_histograms.state,
      num_output_levels.state,
      lower_level.state,
      upper_level.state,
      static_cast<long long>(num_row_pixels),
      static_cast<long long>(num_rows),
      static_cast<long long>(row_stride_samples),
      reinterpret_cast<void*>(stream));

    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_histogram_even(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_histogram_cleanup(cccl_device_histogram_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (build_ptr->jit_compiler)
  {
    delete static_cast<hostjit::JITCompiler*>(build_ptr->jit_compiler);
    build_ptr->jit_compiler = nullptr;
  }
  if (build_ptr->cubin)
  {
    delete[] static_cast<char*>(build_ptr->cubin);
    build_ptr->cubin = nullptr;
  }
  build_ptr->cubin_size   = 0;
  build_ptr->histogram_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_histogram_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
