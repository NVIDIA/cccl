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

// With an init value: 8 args
// (temp_storage, temp_bytes, d_in, d_out, num_items, op_state, init_state, stream)
using reduce_init_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*, void*);

// Without an init value: 7 args
// (temp_storage, temp_bytes, d_in, d_out, num_items, op_state, stream)
using reduce_no_init_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*);

CUresult cccl_device_reduce_build_ex(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_type_info init,
  cccl_init_kind_t init_kind,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  const cccl_build_config* build_config)
try
{
  if (build == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (init_kind == CCCL_FUTURE_VALUE_INIT)
  {
    fprintf(stderr, "\nERROR in cccl_device_reduce_build_ex(): future value init is not supported\n");
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(build_config, cub_path, thrust_path);

  CubCallResult result = [&] {
    auto base = CubCall::from("cub/device/device_reduce.cuh").run("cub::DeviceReduce::Reduce").name("cccl_jit_reduce");

    if (init_kind == CCCL_NO_INIT)
    {
      // cub::DeviceReduce::Reduce with the stateless cub::detail::reduce::no_init tag
      return base.with(temp_storage, temp_bytes, in(input_it), out(output_it), num_items, op, no_init(init), stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
    else // CCCL_VALUE_INIT
    {
      cccl_value_t init_val{init, nullptr}; // state=nullptr; passed at run time
      return base.with(temp_storage, temp_bytes, in(input_it), out(output_it), num_items, op, init_val, stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
  }();

  build->cc             = cc_major * 10 + cc_minor;
  build->cudart_version = cccl::detail::detect_cudart_version(ctk_path);
  cccl::detail::copy_compiled_library(result.compiler, build->payload, build->payload_size);
  build->jit_compiler     = result.compiler;
  build->reduce_fn        = reinterpret_cast<void*>(result.fn_ptr);
  build->accumulator_size = init.size;
  build->determinism      = determinism;
  build->init_kind        = init_kind;

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
  cccl_type_info init,
  cccl_init_kind_t init_kind,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  const cccl_build_config* build_config)
try
{
  if (build == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (init_kind == CCCL_FUTURE_VALUE_INIT)
  {
    fprintf(stderr, "\nERROR in cccl_device_reduce_compile(): future value init is not supported\n");
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(build_config, cub_path, thrust_path);

  auto base = CubCall::from("cub/device/device_reduce.cuh").run("cub::DeviceReduce::Reduce").name("cccl_jit_reduce");

  auto result = [&] {
    if (init_kind == CCCL_NO_INIT)
    {
      return base.with(temp_storage, temp_bytes, in(input_it), out(output_it), num_items, op, no_init(init), stream)
        .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
    else // CCCL_VALUE_INIT
    {
      cccl_value_t init_val{init, nullptr}; // state=nullptr; passed at run time
      return base.with(temp_storage, temp_bytes, in(input_it), out(output_it), num_items, op, init_val, stream)
        .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
  }();

  build->cc             = cc_major * 10 + cc_minor;
  build->cudart_version = cccl::detail::detect_cudart_version(ctk_path);
  cccl::detail::copy_bytes(result.library_bytes, build->payload, build->payload_size);
  // Zero-init fields set by _load, not _compile (matches v1's contract).
  build->jit_compiler     = nullptr;
  build->reduce_fn        = nullptr;
  build->accumulator_size = init.size;
  build->determinism      = determinism;
  build->init_kind        = init_kind;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_compile(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce_load(cccl_device_reduce_build_result_t* build, const char* cudart_path)
try
{
  if (cudart_path == nullptr || cudart_path[0] == '\0' || build == nullptr || build->payload == nullptr
      || build->payload_size == 0 || build->jit_compiler != nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (auto error = cccl::detail::check_device_compute_capability(build->cc); !error.empty())
  {
    fprintf(stderr, "\nERROR in cccl_device_reduce_load(): %s\n", error.c_str());
    return CUDA_ERROR_NO_BINARY_FOR_GPU;
  }

  auto compiler = std::make_unique<hostjit::JITCompiler>(hostjit::CompilerConfig{});
  std::vector<char> library_bytes(
    static_cast<char*>(build->payload), static_cast<char*>(build->payload) + build->payload_size);
  if (!compiler->loadFromBytes(library_bytes, cudart_path, build->cudart_version))
  {
    fprintf(stderr, "\nERROR in cccl_device_reduce_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  auto fn = compiler->getFunction<reduce_init_fn_t>("cccl_jit_reduce");
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

  // The stored fn pointer's arity depends on the build's init kind: CCCL_NO_INIT
  // builds compile the 7-arg wrapper without the init pointer, value-init builds
  // the 8-arg one. Select the matching cast; `init` is ignored for CCCL_NO_INIT.
  if (build.init_kind == CCCL_NO_INIT)
  {
    auto reduce_fn   = reinterpret_cast<reduce_no_init_fn_t>(build.reduce_fn);
    const int status = reduce_fn(
      d_temp_storage, temp_storage_bytes, d_in.state, d_out.state, num_items, op.state, reinterpret_cast<void*>(stream));
    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }

  auto reduce_fn = reinterpret_cast<reduce_init_fn_t>(build.reduce_fn);

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
  cccl_type_info init,
  cccl_init_kind_t init_kind,
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
    init_kind,
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
  if (out_buf == nullptr || out_size == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *out_buf  = nullptr;
  *out_size = 0;

  if (build == nullptr || build->payload == nullptr || build->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // extra = {accumulator_size (u64), determinism (u32), init_kind (u32)}
  std::vector<char> extra =
    cccl::serialization_v2::extra_writer{}
      .put<uint64_t>(build->accumulator_size)
      .put<uint32_t>(build->determinism)
      .put<uint32_t>(build->init_kind)
      .take();

  std::vector<char> blob = cccl::serialization_v2::write_blob(
    CCCL_SERIALIZATION_V2_ALGO_REDUCE,
    build->cc,
    build->cudart_version,
    {"cccl_jit_reduce"},
    extra,
    build->payload,
    build->payload_size);

  cccl::serialization_v2::release_blob(blob, out_buf, out_size);

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
  cccl::serialization_v2::extra_reader extra(parsed.extra);
  const auto accum_size      = extra.get<uint64_t>();
  const auto determinism_raw = extra.get<uint32_t>();
  const auto init_kind_raw   = extra.get<uint32_t>();
  // cccl_device_reduce picks the call arity from init_kind, so reject values
  // it can't dispatch.
  if (!extra.ok() || determinism_raw > CCCL_GPU_TO_GPU
      || (init_kind_raw != CCCL_NO_INIT && init_kind_raw != CCCL_VALUE_INIT))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl_device_reduce_build_result_t result{};
  result.cc             = parsed.cc;
  result.cudart_version = parsed.cudart_version;
  cccl::detail::copy_bytes(
    std::vector<char>(parsed.payload, parsed.payload + parsed.payload_size), result.payload, result.payload_size);
  result.jit_compiler     = nullptr;
  result.reduce_fn        = nullptr;
  result.accumulator_size = accum_size;
  result.determinism      = static_cast<cccl_determinism_t>(determinism_raw);
  result.init_kind        = static_cast<cccl_init_kind_t>(init_kind_raw);

  *build_ptr = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_deserialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
