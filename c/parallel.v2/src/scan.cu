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

#include <cccl/c/scan.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

// Variants with an init value (value or future): 8 args
// (temp, temp_bytes, d_in, d_out, op_state, init_ptr, num_items, stream)
using scan_init_fn_t = int (*)(void*, size_t*, void*, void*, void*, void*, unsigned long long, void*);

// InclusiveScan without init: 7 args
// (temp, temp_bytes, d_in, d_out, op_state, num_items, stream)
using scan_no_init_fn_t = int (*)(void*, size_t*, void*, void*, void*, unsigned long long, void*);

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_scan_build_ex(
  cccl_device_scan_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_type_info init_type,
  bool force_inclusive,
  cccl_init_kind_t init_kind,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  CubCallResult result = [&] {
    auto base = CubCall::from("cub/device/device_scan.cuh").name("cccl_jit_scan");

    if (init_kind == CCCL_NO_INIT)
    {
      // cub::DeviceScan::InclusiveScan(temp, temp_bytes, in, out, op, num_items, stream)
      return base.run("cub::DeviceScan::InclusiveScan")
        .with(temp_storage, temp_bytes, in(d_in), out(d_out), op, num_items, stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
    else if (init_kind == CCCL_VALUE_INIT)
    {
      // ExclusiveScan or InclusiveScanInit with a value init (memcpy'd from void*)
      const char* fn = force_inclusive ? "cub::DeviceScan::InclusiveScanInit" : "cub::DeviceScan::ExclusiveScan";
      cccl_value_t init_val{init_type, nullptr}; // state=nullptr; passed at run time
      return base.run(fn)
        .with(temp_storage, temp_bytes, in(d_in), out(d_out), op, init_val, num_items, stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
    else // CCCL_FUTURE_VALUE_INIT
    {
      // ExclusiveScan or InclusiveScanInit with cub::FutureValue<accum_t>(ptr)
      const char* fn = force_inclusive ? "cub::DeviceScan::InclusiveScanInit" : "cub::DeviceScan::ExclusiveScan";
      return base.run(fn)
        .with(temp_storage, temp_bytes, in(d_in), out(d_out), op, future_val(init_type), num_items, stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
  }();

  build_ptr->cc         = cc_major * 10 + cc_minor;
  build_ptr->cubin      = nullptr;
  build_ptr->cubin_size = 0;
  if (!result.cubin.empty())
  {
    auto* cubin_copy = new char[result.cubin.size()];
    std::memcpy(cubin_copy, result.cubin.data(), result.cubin.size());
    build_ptr->cubin      = cubin_copy;
    build_ptr->cubin_size = result.cubin.size();
  }
  build_ptr->jit_compiler    = result.compiler;
  build_ptr->scan_fn         = result.fn_ptr;
  build_ptr->force_inclusive = force_inclusive;
  build_ptr->init_kind       = init_kind;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_scan_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_scan_build(
  cccl_device_scan_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_type_info init_type,
  bool force_inclusive,
  cccl_init_kind_t init_kind,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_scan_build_ex(
    build_ptr,
    d_in,
    d_out,
    op,
    init_type,
    force_inclusive,
    init_kind,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

// ---------------------------------------------------------------------------
// Run helpers
// ---------------------------------------------------------------------------

static CUresult call_scan_init(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  void* init_ptr, // value state or device pointer for FutureValue
  CUstream stream)
{
  auto fn = reinterpret_cast<scan_init_fn_t>(build.scan_fn);
  if (!fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int status = fn(
    d_temp_storage,
    temp_storage_bytes,
    d_in.state,
    d_out.state,
    op.state,
    init_ptr,
    (unsigned long long) num_items,
    reinterpret_cast<void*>(stream));
  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

CUresult cccl_device_exclusive_scan(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
{
  try
  {
    return call_scan_init(build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init.state, stream);
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_exclusive_scan(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_inclusive_scan(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
{
  try
  {
    return call_scan_init(build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init.state, stream);
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_inclusive_scan(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_exclusive_scan_future_value(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_iterator_t init,
  CUstream stream)
{
  try
  {
    // init.state is the device pointer — passed as void* and wrapped in
    // FutureValue<accum_t> inside the compiled CUDA function.
    return call_scan_init(build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init.state, stream);
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_exclusive_scan_future_value(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_inclusive_scan_future_value(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_iterator_t init,
  CUstream stream)
{
  try
  {
    return call_scan_init(build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init.state, stream);
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_inclusive_scan_future_value(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_inclusive_scan_no_init(
  cccl_device_scan_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  try
  {
    auto fn = reinterpret_cast<scan_no_init_fn_t>(build.scan_fn);
    if (!fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    int status =
      fn(d_temp_storage,
         temp_storage_bytes,
         d_in.state,
         d_out.state,
         op.state,
         (unsigned long long) num_items,
         reinterpret_cast<void*>(stream));
    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_inclusive_scan_no_init(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_scan_cleanup(cccl_device_scan_build_result_t* build_ptr)
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
  build_ptr->cubin_size = 0;
  build_ptr->scan_fn    = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_scan_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
