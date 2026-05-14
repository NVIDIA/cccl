//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include <cccl/c/reduce.h>
#include <hostjit/codegen/cub_call.hpp>

using namespace hostjit::codegen;

using reduce_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*);

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
  // cub_path is an -I prefixed path to the CCCL headers directory;
  // strip the -I prefix to get the bare path for the compiler config.
  const char* cccl_include_path = nullptr;
  std::string cccl_include_str;
  if (libcudacxx_path && libcudacxx_path[0] != '\0')
  {
    cccl_include_str = libcudacxx_path;
    if (cccl_include_str.substr(0, 2) == "-I")
    {
      cccl_include_str = cccl_include_str.substr(2);
    }
    cccl_include_path = cccl_include_str.c_str();
  }

  // ctk_path is an -I prefixed path to the CTK include directory;
  // strip the -I prefix and /include suffix to get the toolkit root.
  const char* ctk_root = nullptr;
  std::string ctk_root_str;
  if (ctk_path && ctk_path[0] != '\0')
  {
    ctk_root_str = ctk_path;
    if (ctk_root_str.substr(0, 2) == "-I")
    {
      ctk_root_str = ctk_root_str.substr(2);
    }
    // The Python layer passes the include directory itself; the C++ config
    // expects the toolkit root (parent of include/).
    // Walk up from the include dir until we find the directory containing
    // nvvm/libdevice/ — that is the real toolkit root.  This handles both
    //   /usr/local/cuda/include           -> /usr/local/cuda
    //   /usr/local/cuda/targets/.../include -> /usr/local/cuda
    std::filesystem::path p(ctk_root_str);
    if (p.filename() == "include")
    {
      p = p.parent_path();
    }
    for (auto candidate = p; candidate.has_parent_path() && candidate != candidate.parent_path();
         candidate      = candidate.parent_path())
    {
      if (std::filesystem::exists(candidate / "nvvm" / "libdevice"))
      {
        p = candidate;
        break;
      }
    }
    ctk_root_str = p.string();
    ctk_root     = ctk_root_str.c_str();
  }

  // Collect any extra -I paths from the legacy cub_path / thrust_path arguments.
  std::vector<std::string> extra_include_strs;
  std::vector<const char*> extra_include_ptrs;
  for (const char* path : {cub_path, thrust_path})
  {
    if (path && path[0] != '\0')
    {
      std::string s = path;
      if (s.substr(0, 2) == "-I")
      {
        s = s.substr(2);
      }
      extra_include_strs.push_back(std::move(s));
    }
  }
  for (const auto& s : extra_include_strs)
  {
    extra_include_ptrs.push_back(s.c_str());
  }

  // Merge with any user-provided build config.
  cccl_build_config merged_config{};
  if (build_config)
  {
    merged_config = *build_config;
  }
  // Append legacy include dirs to any existing extra_include_dirs.
  std::vector<const char*> all_include_ptrs;
  for (size_t i = 0; i < merged_config.num_extra_include_dirs; ++i)
  {
    all_include_ptrs.push_back(merged_config.extra_include_dirs[i]);
  }
  all_include_ptrs.insert(all_include_ptrs.end(), extra_include_ptrs.begin(), extra_include_ptrs.end());
  merged_config.extra_include_dirs     = all_include_ptrs.data();
  merged_config.num_extra_include_dirs = all_include_ptrs.size();

  auto result =
    CubCall::from("cub/device/device_reduce.cuh")
      .run("cub::DeviceReduce::Reduce")
      .name("cccl_jit_reduce")
      .with(temp_storage, temp_bytes, in(input_it), out(output_it), num_items, op, init)
      .compile(cc_major, cc_minor, &merged_config, ctk_root, cccl_include_path);

  build->cc         = cc_major * 10 + cc_minor;
  build->cubin      = nullptr;
  build->cubin_size = 0;
  if (!result.cubin.empty())
  {
    auto* cubin_copy = new char[result.cubin.size()];
    std::memcpy(cubin_copy, result.cubin.data(), result.cubin.size());
    build->cubin      = cubin_copy;
    build->cubin_size = result.cubin.size();
  }
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

CUresult cccl_device_reduce(
  cccl_device_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream /*stream*/)
{
  try
  {
    auto reduce_fn = reinterpret_cast<reduce_fn_t>(build.reduce_fn);

    if (!reduce_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    // Parameter order matches CubCall::with() order: ..., num_items, op.state, init.state
    int status =
      reduce_fn(d_temp_storage, temp_storage_bytes, d_in.state, d_out.state, num_items, op.state, init.state);

    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_reduce(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
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
  build_ptr->reduce_fn  = nullptr;

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
