//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//
//
// cuda.compute scan-by-key, HostJIT/v2 backend.
//
// The semantics come straight from the public CUB overloads, which differ from the plain scan in two
// ways that the wrapper must not paper over:
//   * `InclusiveScanByKey` has NO init parameter, so `force_inclusive` together with an init value is
//     rejected rather than silently mapped onto some other overload.
//   * `ExclusiveScanByKey` restarts the init value at the head of every key run; the equality operator
//     decides run boundaries and is therefore a separate operator instance from the scan operator.

#include <cstdio>
#include <cstring>

#include <cccl/c/scan_by_key.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>

using namespace hostjit::codegen;

// InclusiveScanByKey: 9 args
// (temp, temp_bytes, keys_in, values_in, values_out, scan_op, num_items, equality_op, stream)
using scan_by_key_no_init_fn_t = int (*)(void*, size_t*, void*, void*, void*, void*, unsigned long long, void*, void*);

// ExclusiveScanByKey: 10 args
// (temp, temp_bytes, keys_in, values_in, values_out, scan_op, init_ptr, num_items, equality_op, stream)
using scan_by_key_init_fn_t =
  int (*)(void*, size_t*, void*, void*, void*, void*, void*, unsigned long long, void*, void*);

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_scan_by_key_build_ex(
  cccl_device_scan_by_key_build_result_t* build_ptr,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t op,
  cccl_op_t equality_op,
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
  // CUB's inclusive by-key scan takes no init value; there is nothing to bind it to.
  if (force_inclusive && init_kind != CCCL_NO_INIT)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  CubCallResult result = [&] {
    auto base = CubCall::from("cub/device/device_scan.cuh").name("cccl_jit_scan_by_key");

    if (init_kind == CCCL_NO_INIT)
    {
      // cub::DeviceScan::InclusiveScanByKey(temp, bytes, keys_in, values_in, values_out, op,
      //                                     num_items, equality_op, stream)
      return base.run("cub::DeviceScan::InclusiveScanByKey")
        .with(force_accum_type(d_values_in.value_type),
              temp_storage,
              temp_bytes,
              in(d_keys_in),
              in(d_values_in),
              out(d_values_out),
              op,
              num_items,
              cmp(equality_op),
              stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }

    // ExclusiveScanByKey with a value init (memcpy'd from void* at run time).
    cccl_value_t init_val{init_type, nullptr};
    return base.run("cub::DeviceScan::ExclusiveScanByKey")
      .with(force_accum_type(d_values_in.value_type),
            temp_storage,
            temp_bytes,
            in(d_keys_in),
            in(d_values_in),
            out(d_values_out),
            op,
            init_val,
            num_items,
            cmp(equality_op),
            stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
  }();

  build_ptr->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_cubin(result.cubin, build_ptr->payload, build_ptr->payload_size);
  build_ptr->jit_compiler    = result.compiler;
  build_ptr->scan_by_key_fn  = result.fn_ptr;
  build_ptr->force_inclusive = force_inclusive;
  build_ptr->init_kind       = init_kind;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_scan_by_key_build_ex(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_scan_by_key_build(
  cccl_device_scan_by_key_build_result_t* build_ptr,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t op,
  cccl_op_t equality_op,
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
  return cccl_device_scan_by_key_build_ex(
    build_ptr,
    d_keys_in,
    d_values_in,
    d_values_out,
    op,
    equality_op,
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
// Run
// ---------------------------------------------------------------------------

CUresult cccl_device_inclusive_scan_by_key(
  cccl_device_scan_by_key_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_op_t equality_op,
  CUstream stream)
try
{
  if (!build.scan_by_key_fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // ABI guard: an init-bearing build result holds the 10-arg entry point.
  if (build.init_kind != CCCL_NO_INIT)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto fn          = reinterpret_cast<scan_by_key_no_init_fn_t>(build.scan_by_key_fn);
  const int status = fn(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in.state,
    d_values_in.state,
    d_values_out.state,
    op.state,
    static_cast<unsigned long long>(num_items),
    equality_op.state,
    reinterpret_cast<void*>(stream));
  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_inclusive_scan_by_key(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_exclusive_scan_by_key(
  cccl_device_scan_by_key_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_op_t equality_op,
  cccl_value_t init,
  CUstream stream)
try
{
  if (!build.scan_by_key_fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // ABI guard: the no-init build result holds the 9-arg entry point.
  if (build.init_kind == CCCL_NO_INIT)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto fn          = reinterpret_cast<scan_by_key_init_fn_t>(build.scan_by_key_fn);
  const int status = fn(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in.state,
    d_values_in.state,
    d_values_out.state,
    op.state,
    init.state,
    static_cast<unsigned long long>(num_items),
    equality_op.state,
    reinterpret_cast<void*>(stream));
  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_exclusive_scan_by_key(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_scan_by_key_cleanup(cccl_device_scan_by_key_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->scan_by_key_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_scan_by_key_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
