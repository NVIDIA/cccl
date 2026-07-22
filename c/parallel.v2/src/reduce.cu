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
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cccl/c/reduce.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>
#include <util/serialization.h>

using namespace hostjit::codegen;

// (temp_storage, temp_bytes, d_in, d_out, num_items, op_state, init_state, stream)
using reduce_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*, void*);

CUresult cccl_device_reduce_build_ex(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_value_t init,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* build_config)
try
{
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(build_config, cub_path, thrust_path);

  auto result =
    CubCall::from("cub/device/device_reduce.cuh")
      .run("cub::DeviceReduce::Reduce")
      .name("cccl_jit_reduce")
      .with(temp_storage, temp_bytes, in(input_it), out(output_it), num_items, op, init, stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build->cc = cc_major * 10 + cc_minor;
  // Populate payload with the compiled artifact bytes too (not just the
  // live loaded state) so cccl_device_reduce_serialize works uniformly
  // whether the build_result came from this fused path or from
  // _compile()+_load(). Best-effort: a failed read leaves payload null,
  // which only affects later serialize() calls, not this build itself.
  auto library_bytes = cccl::detail::read_compiled_library_bytes(result.compiler);
  cccl::detail::copy_bytes(library_bytes, build->payload, build->payload_size);
  build->jit_compiler     = result.compiler;
  build->reduce_fn        = reinterpret_cast<void*>(result.fn_ptr);
  build->accumulator_size = init.type.size;
  build->determinism      = determinism;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_compile(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_value_t init,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* build_config)
try
{
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(build_config, cub_path, thrust_path);

  auto result =
    CubCall::from("cub/device/device_reduce.cuh")
      .run("cub::DeviceReduce::Reduce")
      .name("cccl_jit_reduce")
      .with(temp_storage, temp_bytes, in(input_it), out(output_it), num_items, op, init, stream)
      .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_bytes(result.library_bytes, build->payload, build->payload_size);
  // Zero-init fields set by _load, not _compile (matches v1's contract).
  build->jit_compiler     = nullptr;
  build->reduce_fn        = nullptr;
  build->accumulator_size = init.type.size;
  build->determinism      = determinism;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_compile(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_load(cccl_device_reduce_build_result_t* build, const char* ctk_path)
try
{
  if (build == nullptr || build->payload == nullptr || build->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  hostjit::CompilerConfig jit_config = cccl::detail::make_load_jit_config(ctk_path);

  auto compiler = std::make_unique<hostjit::JITCompiler>(jit_config);
  std::vector<char> library_bytes(
    static_cast<char*>(build->payload), static_cast<char*>(build->payload) + build->payload_size);
  if (!compiler->loadFromBytes(library_bytes))
  {
    fprintf(stderr, "\nERROR in cccl_device_reduce_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  auto fn = compiler->getFunction<reduce_fn_t>("cccl_jit_reduce");
  if (!fn)
  {
    fprintf(stderr, "\nERROR in cccl_device_reduce_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  build->jit_compiler = compiler.release();
  build->reduce_fn    = reinterpret_cast<void*>(fn);

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_load(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce(
  cccl_device_reduce_build_result_t build,
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
  if (!build.reduce_fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto reduce_fn = reinterpret_cast<reduce_fn_t>(build.reduce_fn);

  // Parameter order matches CubCall::with() order: ..., num_items, op.state, init.state, stream
  const int status = reduce_fn(
    d_temp_storage,
    temp_storage_bytes,
    d_in.state,
    d_out.state,
    num_items,
    op.state,
    init.state,
    reinterpret_cast<void*>(stream));

  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_nondeterministic(
  cccl_device_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
{
  return cccl_device_reduce(build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init, stream);
}

CUresult cccl_device_reduce_cleanup(cccl_device_reduce_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->reduce_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_build(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_value_t init,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_reduce_build_ex(
    build,
    d_in,
    d_out,
    op,
    init,
    determinism,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_reduce_serialize(const cccl_device_reduce_build_result_t* build, void** out_buf, size_t* out_size)
try
{
  *out_buf  = nullptr;
  *out_size = 0;

  if (build == nullptr || build->payload == nullptr || build->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // extra = {accumulator_size (u64), determinism (u32)}
  std::vector<char> extra(sizeof(uint64_t) + sizeof(uint32_t));
  uint64_t accum_size_le  = build->accumulator_size;
  uint32_t determinism_le = static_cast<uint32_t>(build->determinism);
  std::memcpy(extra.data(), &accum_size_le, sizeof(accum_size_le));
  std::memcpy(extra.data() + sizeof(accum_size_le), &determinism_le, sizeof(determinism_le));

  std::vector<char> blob = cccl::serialization_v2::write_blob(
    CCCL_SERIALIZATION_V2_ALGO_REDUCE, build->cc, {"cccl_jit_reduce"}, extra, build->payload, build->payload_size);

  auto* buf = new char[blob.size()];
  std::memcpy(buf, blob.data(), blob.size());
  *out_buf  = buf;
  *out_size = blob.size();

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_serialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_deserialize(cccl_device_reduce_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::serialization_v2::parsed_blob parsed =
    cccl::serialization_v2::read_blob(CCCL_SERIALIZATION_V2_ALGO_REDUCE, buf, size);

  // Commit-on-success: build a local result{} and only assign to
  // *build_ptr after every read succeeds, so *build_ptr is left unchanged
  // on failure (matches v1's deserialize contract).
  if (parsed.extra.size() != sizeof(uint64_t) + sizeof(uint32_t))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  uint64_t accum_size;
  uint32_t determinism_raw;
  std::memcpy(&accum_size, parsed.extra.data(), sizeof(accum_size));
  std::memcpy(&determinism_raw, parsed.extra.data() + sizeof(accum_size), sizeof(determinism_raw));

  cccl_device_reduce_build_result_t result{};
  result.cc = parsed.cc;
  cccl::detail::copy_bytes(
    std::vector<char>(parsed.payload, parsed.payload + parsed.payload_size), result.payload, result.payload_size);
  result.jit_compiler     = nullptr;
  result.reduce_fn        = nullptr;
  result.accumulator_size = accum_size;
  result.determinism      = static_cast<cccl_determinism_t>(determinism_raw);

  *build_ptr = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_deserialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
