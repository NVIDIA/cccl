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

#include <cccl/c/scan.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>
#include <util/serialization.h>

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
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
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

  build_ptr->cc      = cc_major * 10 + cc_minor;
  auto library_bytes = cccl::detail::read_compiled_library_bytes(result.compiler);
  cccl::detail::copy_bytes(library_bytes, build_ptr->payload, build_ptr->payload_size);
  build_ptr->jit_compiler    = result.compiler;
  build_ptr->scan_fn         = result.fn_ptr;
  build_ptr->force_inclusive = force_inclusive;
  build_ptr->init_kind       = init_kind;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_scan_build_ex(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_scan_compile(
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
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  CubCallCompileOnlyResult result = [&] {
    auto base = CubCall::from("cub/device/device_scan.cuh").name("cccl_jit_scan");

    if (init_kind == CCCL_NO_INIT)
    {
      return base.run("cub::DeviceScan::InclusiveScan")
        .with(temp_storage, temp_bytes, in(d_in), out(d_out), op, num_items, stream)
        .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
    else if (init_kind == CCCL_VALUE_INIT)
    {
      const char* fn = force_inclusive ? "cub::DeviceScan::InclusiveScanInit" : "cub::DeviceScan::ExclusiveScan";
      cccl_value_t init_val{init_type, nullptr};
      return base.run(fn)
        .with(temp_storage, temp_bytes, in(d_in), out(d_out), op, init_val, num_items, stream)
        .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
    else // CCCL_FUTURE_VALUE_INIT
    {
      const char* fn = force_inclusive ? "cub::DeviceScan::InclusiveScanInit" : "cub::DeviceScan::ExclusiveScan";
      return base.run(fn)
        .with(temp_storage, temp_bytes, in(d_in), out(d_out), op, future_val(init_type), num_items, stream)
        .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
  }();

  build_ptr->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_bytes(result.library_bytes, build_ptr->payload, build_ptr->payload_size);
  build_ptr->jit_compiler    = nullptr;
  build_ptr->scan_fn         = nullptr;
  build_ptr->force_inclusive = force_inclusive;
  build_ptr->init_kind       = init_kind;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_scan_compile(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_scan_load(cccl_device_scan_build_result_t* build_ptr, const char* ctk_path)
try
{
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  hostjit::CompilerConfig jit_config = cccl::detail::make_load_jit_config(ctk_path);
  auto compiler                      = std::make_unique<hostjit::JITCompiler>(jit_config);
  std::vector<char> library_bytes(
    static_cast<char*>(build_ptr->payload), static_cast<char*>(build_ptr->payload) + build_ptr->payload_size);
  if (!compiler->loadFromBytes(library_bytes))
  {
    fprintf(stderr, "\nERROR in cccl_device_scan_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }
  // Both scan_init_fn_t and scan_no_init_fn_t are extern "C" (void*, ...)
  // functions exported under the same fixed name; only the arg count
  // differs (dispatched at call time via build.init_kind), so a raw
  // function-pointer getFunction<void*> works for either.
  using generic_fn_t = void*;
  auto fn            = compiler->getFunction<generic_fn_t>("cccl_jit_scan");
  if (!fn)
  {
    fprintf(stderr, "\nERROR in cccl_device_scan_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  build_ptr->jit_compiler = compiler.release();
  build_ptr->scan_fn      = fn;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_scan_load(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_scan_serialize(const cccl_device_scan_build_result_t* build_ptr, void** out_buf, size_t* out_size)
try
{
  *out_buf  = nullptr;
  *out_size = 0;
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // extra = {force_inclusive (u8), init_kind (u32)}
  std::vector<char> extra(sizeof(uint8_t) + sizeof(uint32_t));
  uint8_t force_inclusive_u8 = build_ptr->force_inclusive ? 1 : 0;
  uint32_t init_kind_u32     = static_cast<uint32_t>(build_ptr->init_kind);
  std::memcpy(extra.data(), &force_inclusive_u8, sizeof(force_inclusive_u8));
  std::memcpy(extra.data() + sizeof(force_inclusive_u8), &init_kind_u32, sizeof(init_kind_u32));

  std::vector<char> blob = cccl::serialization_v2::write_blob(
    CCCL_SERIALIZATION_V2_ALGO_SCAN,
    build_ptr->cc,
    {"cccl_jit_scan"},
    extra,
    build_ptr->payload,
    build_ptr->payload_size);
  auto* buf = new char[blob.size()];
  std::memcpy(buf, blob.data(), blob.size());
  *out_buf  = buf;
  *out_size = blob.size();
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_scan_serialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_scan_deserialize(cccl_device_scan_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cccl::serialization_v2::parsed_blob parsed =
    cccl::serialization_v2::read_blob(CCCL_SERIALIZATION_V2_ALGO_SCAN, buf, size);
  if (parsed.extra.size() != sizeof(uint8_t) + sizeof(uint32_t))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  uint8_t force_inclusive_u8;
  uint32_t init_kind_u32;
  std::memcpy(&force_inclusive_u8, parsed.extra.data(), sizeof(force_inclusive_u8));
  std::memcpy(&init_kind_u32, parsed.extra.data() + sizeof(force_inclusive_u8), sizeof(init_kind_u32));

  cccl_device_scan_build_result_t result{};
  result.cc = parsed.cc;
  cccl::detail::copy_bytes(
    std::vector<char>(parsed.payload, parsed.payload + parsed.payload_size), result.payload, result.payload_size);
  result.jit_compiler    = nullptr;
  result.scan_fn         = nullptr;
  result.force_inclusive = force_inclusive_u8 != 0;
  result.init_kind       = static_cast<cccl_init_kind_t>(init_kind_u32);

  *build_ptr = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_scan_deserialize(): %s\n", exc.what());
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
  if (!build.scan_fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // Guard against ABI mismatch: this path uses the 8-arg scan_init_fn_t
  // (with init pointer). Calling it with a build result compiled for
  // CCCL_NO_INIT (7-arg scan_no_init_fn_t) would be undefined behaviour.
  if (build.init_kind == CCCL_NO_INIT)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto fn          = reinterpret_cast<scan_init_fn_t>(build.scan_fn);
  const int status = fn(
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
try
{
  return call_scan_init(build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init.state, stream);
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_exclusive_scan(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
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
try
{
  return call_scan_init(build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init.state, stream);
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_inclusive_scan(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
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
try
{
  return call_scan_init(build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init.state, stream);
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_inclusive_scan_future_value(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
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
try
{
  if (!build.scan_fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // Guard against ABI mismatch: this path uses the 7-arg scan_no_init_fn_t.
  // A build result compiled with an init value stores an 8-arg scan_init_fn_t;
  // casting it here would be undefined behaviour.
  if (build.init_kind != CCCL_NO_INIT)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto fn = reinterpret_cast<scan_no_init_fn_t>(build.scan_fn);
  const int status =
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
  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->scan_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_scan_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
