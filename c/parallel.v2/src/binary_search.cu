//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/version>

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <cccl/c/binary_search.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>
#include <util/first_call_gate.h>
#include <util/serialization.h>

using namespace hostjit::codegen;

// (d_in_0, num_items, d_in_1, num_values, d_out_0, op_0_state, stream)
using binary_search_fn_t = int (*)(void*, unsigned long long, void*, unsigned long long, void*, void*, void*);

CUresult cccl_device_binary_search_build_ex(
  cccl_device_binary_search_build_result_t* build_ptr,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
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
#if CCCL_OS(WINDOWS)
  build_ptr->first_call_state = nullptr;
#endif
  const std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  const std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* const cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* const ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);
#if CCCL_OS(WINDOWS)
  auto first_call_state = std::make_unique<cccl::detail::first_call_gate>();
#endif

  const char* find_fn =
    (mode == CCCL_BINARY_SEARCH_LOWER_BOUND) ? "cub::DeviceFind::LowerBound" : "cub::DeviceFind::UpperBound";

  // env_stream uses the env-based DeviceFind overload so CUB manages its own
  // temp storage via the env's memory_resource — no caller-managed buffer.
  auto result =
    CubCall::from("cub/device/device_find.cuh")
      .run(find_fn)
      .name("cccl_jit_binary_search")
      .with(in(d_data), num_haystack, in(d_values), num_needles, out(d_out), cmp(op), env_stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build_ptr->cc = cc_major * 10 + cc_minor;
  // Populate payload with the compiled artifact bytes too (not just the
  // live loaded state) so cccl_device_binary_search_serialize works
  // uniformly whether the build_result came from this fused path or from
  // _compile()+_load(). Best-effort: a failed read leaves payload null,
  // which only affects later serialize() calls, not this build itself.
  auto library_bytes = cccl::detail::read_compiled_library_bytes(result.compiler);
  cccl::detail::copy_bytes(library_bytes, build_ptr->payload, build_ptr->payload_size);
  build_ptr->jit_compiler = result.compiler;
#if CCCL_OS(WINDOWS)
  build_ptr->first_call_state = first_call_state.release();
#endif
  build_ptr->binary_search_fn = result.fn_ptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search_build_ex(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_binary_search_compile(
  cccl_device_binary_search_build_result_t* build_ptr,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
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
#if CCCL_OS(WINDOWS)
  build_ptr->first_call_state = nullptr;
#endif
  const std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  const std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* const cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* const ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  const char* find_fn =
    (mode == CCCL_BINARY_SEARCH_LOWER_BOUND) ? "cub::DeviceFind::LowerBound" : "cub::DeviceFind::UpperBound";

  // env_stream uses the env-based DeviceFind overload so CUB manages its own
  // temp storage via the env's memory_resource — no caller-managed buffer.
  auto result =
    CubCall::from("cub/device/device_find.cuh")
      .run(find_fn)
      .name("cccl_jit_binary_search")
      .with(in(d_data), num_haystack, in(d_values), num_needles, out(d_out), cmp(op), env_stream)
      .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build_ptr->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_bytes(result.library_bytes, build_ptr->payload, build_ptr->payload_size);
  // Zero-init fields set by _load, not _compile (matches v1's contract).
  build_ptr->jit_compiler     = nullptr;
  build_ptr->binary_search_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search_compile(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_binary_search_load(cccl_device_binary_search_build_result_t* build_ptr, const char* ctk_path)
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
    fprintf(stderr, "\nERROR in cccl_device_binary_search_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }
  auto fn = compiler->getFunction<binary_search_fn_t>("cccl_jit_binary_search");
  if (!fn)
  {
    fprintf(stderr, "\nERROR in cccl_device_binary_search_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  build_ptr->jit_compiler = compiler.release();
#if CCCL_OS(WINDOWS)
  build_ptr->first_call_state = new cccl::detail::first_call_gate();
#endif
  build_ptr->binary_search_fn = reinterpret_cast<void*>(fn);

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search_load(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_binary_search_serialize(
  const cccl_device_binary_search_build_result_t* build_ptr, void** out_buf, size_t* out_size)
try
{
  *out_buf  = nullptr;
  *out_size = 0;
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::vector<char> blob = cccl::serialization_v2::write_blob(
    CCCL_SERIALIZATION_V2_ALGO_BINARY_SEARCH,
    build_ptr->cc,
    {"cccl_jit_binary_search"},
    /*extra=*/{},
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
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search_serialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult
cccl_device_binary_search_deserialize(cccl_device_binary_search_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cccl::serialization_v2::parsed_blob parsed =
    cccl::serialization_v2::read_blob(CCCL_SERIALIZATION_V2_ALGO_BINARY_SEARCH, buf, size);

  cccl_device_binary_search_build_result_t result{};
  result.cc = parsed.cc;
  cccl::detail::copy_bytes(
    std::vector<char>(parsed.payload, parsed.payload + parsed.payload_size), result.payload, result.payload_size);
  result.jit_compiler = nullptr;
#if CCCL_OS(WINDOWS)
  result.first_call_state = nullptr;
#endif
  result.binary_search_fn = nullptr;

  *build_ptr = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search_deserialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_binary_search(
  cccl_device_binary_search_build_result_t build,
  cccl_iterator_t d_data,
  uint64_t num_items,
  cccl_iterator_t d_values,
  uint64_t num_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  CUstream stream)
try
{
  if (!build.binary_search_fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const auto fn = reinterpret_cast<binary_search_fn_t>(build.binary_search_fn);

#if CCCL_OS(WINDOWS)
  if (!build.first_call_state)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const auto invoke = [&] {
    return fn(
      d_data.state, num_items, d_values.state, num_values, d_out.state, op.state, reinterpret_cast<void*>(stream));
  };
  // Empty calls return before DeviceTransform initializes its static launch configuration,
  // so they must not complete the first-call gate.
  const int status =
    num_values == 0 ? invoke() : static_cast<cccl::detail::first_call_gate*>(build.first_call_state)->invoke(invoke);
#else
  const int status =
    fn(d_data.state, num_items, d_values.state, num_values, d_out.state, op.state, reinterpret_cast<void*>(stream));
#endif
  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_binary_search_build(
  cccl_device_binary_search_build_result_t* build,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_binary_search_build_ex(
    build,
    mode,
    d_data,
    d_values,
    d_out,
    op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_binary_search_cleanup(cccl_device_binary_search_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

#if CCCL_OS(WINDOWS)
  delete static_cast<cccl::detail::first_call_gate*>(build_ptr->first_call_state);
  build_ptr->first_call_state = nullptr;
#endif
  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->binary_search_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
