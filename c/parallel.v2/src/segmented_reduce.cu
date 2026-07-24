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
#include <string>
#include <vector>

#include <cccl/c/segmented_reduce.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>
#include <util/serialization.h>

using namespace hostjit::codegen;

// (temp_storage, temp_bytes, d_in, d_out, num_segments, begin_offsets, end_offsets, op, init, stream)
using segmented_reduce_fn_t =
  int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*, void*, void*, void*);

CUresult cccl_device_segmented_reduce_build_ex(
  cccl_device_segmented_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_iterator_t start_offset_it,
  cccl_iterator_t end_offset_it,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* build_config)
try
{
  const std::string cccl_include_str = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  const char* cccl_include_path      = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();

  const std::string ctk_root_str = cccl::detail::parse_ctk_root(ctk_path);
  const char* ctk_root           = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(build_config, cub_path, thrust_path);

  auto result =
    CubCall::from("cub/device/device_segmented_reduce.cuh")
      .run("cub::DeviceSegmentedReduce::Reduce")
      .name("cccl_jit_segmented_reduce")
      .with(temp_storage,
            temp_bytes,
            in(d_in),
            out(d_out),
            num_items,
            in(start_offset_it),
            in(end_offset_it),
            op,
            init,
            stream)
      .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build->cc          = cc_major * 10 + cc_minor;
  auto library_bytes = cccl::detail::read_compiled_library_bytes(result.compiler);
  cccl::detail::copy_bytes(library_bytes, build->payload, build->payload_size);
  build->jit_compiler        = result.compiler;
  build->segmented_reduce_fn = reinterpret_cast<void*>(result.fn_ptr);

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_reduce_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_reduce_compile(
  cccl_device_segmented_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_iterator_t start_offset_it,
  cccl_iterator_t end_offset_it,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* build_config)
try
{
  const std::string cccl_include_str = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  const char* cccl_include_path      = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();

  const std::string ctk_root_str = cccl::detail::parse_ctk_root(ctk_path);
  const char* ctk_root           = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(build_config, cub_path, thrust_path);

  auto result =
    CubCall::from("cub/device/device_segmented_reduce.cuh")
      .run("cub::DeviceSegmentedReduce::Reduce")
      .name("cccl_jit_segmented_reduce")
      .with(temp_storage,
            temp_bytes,
            in(d_in),
            out(d_out),
            num_items,
            in(start_offset_it),
            in(end_offset_it),
            op,
            init,
            stream)
      .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_bytes(result.library_bytes, build->payload, build->payload_size);
  build->jit_compiler        = nullptr;
  build->segmented_reduce_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_reduce_compile(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_reduce_load(cccl_device_segmented_reduce_build_result_t* build, const char* ctk_path)
try
{
  if (build == nullptr || build->payload == nullptr || build->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  hostjit::CompilerConfig jit_config = cccl::detail::make_load_jit_config(ctk_path);
  auto compiler                      = std::make_unique<hostjit::JITCompiler>(jit_config);
  std::vector<char> library_bytes(
    static_cast<char*>(build->payload), static_cast<char*>(build->payload) + build->payload_size);
  if (!compiler->loadFromBytes(library_bytes))
  {
    fprintf(stderr, "\nERROR in cccl_device_segmented_reduce_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  auto fn = compiler->getFunction<segmented_reduce_fn_t>("cccl_jit_segmented_reduce");
  if (!fn)
  {
    fprintf(stderr, "\nERROR in cccl_device_segmented_reduce_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  build->jit_compiler        = compiler.release();
  build->segmented_reduce_fn = reinterpret_cast<void*>(fn);

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_reduce_load(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_reduce_serialize(
  const cccl_device_segmented_reduce_build_result_t* build, void** out_buf, size_t* out_size)
try
{
  *out_buf  = nullptr;
  *out_size = 0;
  if (build == nullptr || build->payload == nullptr || build->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::vector<char> blob = cccl::serialization_v2::write_blob(
    CCCL_SERIALIZATION_V2_ALGO_SEGMENTED_REDUCE,
    build->cc,
    {"cccl_jit_segmented_reduce"},
    /*extra=*/{},
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
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_reduce_serialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_reduce_deserialize(
  cccl_device_segmented_reduce_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::serialization_v2::parsed_blob parsed =
    cccl::serialization_v2::read_blob(CCCL_SERIALIZATION_V2_ALGO_SEGMENTED_REDUCE, buf, size);

  cccl_device_segmented_reduce_build_result_t result{};
  result.cc = parsed.cc;
  cccl::detail::copy_bytes(
    std::vector<char>(parsed.payload, parsed.payload + parsed.payload_size), result.payload, result.payload_size);
  result.jit_compiler        = nullptr;
  result.segmented_reduce_fn = nullptr;

  *build_ptr = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_reduce_deserialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_reduce(
  cccl_device_segmented_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_segments,
  cccl_iterator_t start_offset,
  cccl_iterator_t end_offset,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
try
{
  if (!build.segmented_reduce_fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto segmented_reduce_fn = reinterpret_cast<segmented_reduce_fn_t>(build.segmented_reduce_fn);

  // Parameter order matches CubCall::with() order:
  // temp_storage, temp_bytes, d_in, d_out, num_items, begin_offsets, end_offsets, op, init, stream
  const int status = segmented_reduce_fn(
    d_temp_storage,
    temp_storage_bytes,
    d_in.state,
    d_out.state,
    num_segments,
    start_offset.state,
    end_offset.state,
    op.state,
    init.state,
    reinterpret_cast<void*>(stream));

  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_reduce(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_segmented_reduce_build(
  cccl_device_segmented_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_iterator_t begin_offset_in,
  cccl_iterator_t end_offset_in,
  cccl_op_t op,
  cccl_value_t init,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_segmented_reduce_build_ex(
    build,
    d_in,
    d_out,
    begin_offset_in,
    end_offset_in,
    op,
    init,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_segmented_reduce_cleanup(cccl_device_segmented_reduce_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->segmented_reduce_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_segmented_reduce_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
