//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>

#include <cccl/c/binary_search.h>
#include <hostjit/codegen/bitcode.hpp>
#include <hostjit/codegen/iterators.hpp>
#include <hostjit/codegen/operators.hpp>
#include <hostjit/codegen/types.hpp>
#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>
#include <util/build_utils.h>

using namespace hostjit;
using namespace hostjit::codegen;

// d_data_state, num_items, d_values_state, num_values, d_out_state, op_state
using binary_search_fn_t = int (*)(void*, unsigned long long, void*, unsigned long long, void*, void*);

static std::string make_binary_search_source(
  cccl_iterator_t d_data, cccl_iterator_t d_values, cccl_iterator_t d_out, cccl_op_t op, cccl_binary_search_mode_t mode)
{
  const auto data_type   = get_type_name(d_data.value_type.type);
  const auto values_type = get_type_name(d_values.value_type.type);
  const auto out_type    = get_type_name(d_out.value_type.type);
  const bool has_bc      = BitcodeCollector::is_bitcode_op(op);

  auto data_code   = make_input_iterator(d_data, data_type, data_type, "in_0_it_t", "in_0", "d_in_0");
  auto values_code = make_input_iterator(d_values, values_type, values_type, "in_1_it_t", "in_1", "d_in_1");
  auto out_code    = make_output_iterator(d_out, out_type, "out_0_it_t", "out_0", "d_out_0");
  auto op_code     = make_comparison_op(op, data_type, "CompareOp", "op_0", "op_0_state", has_bc);

  const std::string mode_str =
    (mode == CCCL_BINARY_SEARCH_LOWER_BOUND) ? "cub::detail::find::lower_bound" : "cub::detail::find::upper_bound";

  std::string src = R"(#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/__iterator/zip_iterator.h>
#include <cub/agent/agent_for.cuh>
#include <cub/detail/binary_search_helpers.cuh>
#include <climits>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

)";

  src += data_code.preamble;
  src += values_code.preamble;
  src += out_code.preamble;
  src += op_code.preamble;

  src += R"(using OffsetT = unsigned long long;
using policy_dim_t = cub::detail::for_each::policy_t<256, 2>;
struct device_for_policy {
  struct ActivePolicy {
    using for_policy_t = policy_dim_t;
  };
};

)";

  // Template kernel — types deduced when called with <<< >>>
  src += std::format(
    R"(template<typename DataIt, typename ValuesIt, typename OutIt, typename CompOp>
_CCCL_KERNEL_ATTRIBUTES
__launch_bounds__(device_for_policy::ActivePolicy::for_policy_t::threads_per_block)
void binary_search_kernel(DataIt d_data, OffsetT num_data, ValuesIt d_values, OffsetT num_values, OutIt d_out, CompOp op)
{{
  auto input_it     = cuda::make_zip_iterator(d_values, d_out);
  auto comp_wrapper = cub::detail::find::make_comp_wrapper<{}>(d_data, num_data, op);
  auto agent_op     = [&comp_wrapper, &input_it](OffsetT index) {{
    comp_wrapper(input_it[index]);
  }};
  using active_policy_t = device_for_policy::ActivePolicy::for_policy_t;
  using agent_t = cub::detail::for_each::agent_block_striped_t<active_policy_t, OffsetT, decltype(agent_op)>;
  constexpr auto threads_per_block  = active_policy_t::threads_per_block;
  constexpr auto items_per_tile = active_policy_t::items_per_thread * threads_per_block;
  const auto tile_base     = static_cast<OffsetT>(blockIdx.x) * items_per_tile;
  const auto num_remaining = num_values - tile_base;
  const auto items_in_tile = static_cast<OffsetT>(num_remaining < items_per_tile ? num_remaining : items_per_tile);
  if (items_in_tile == items_per_tile) {{
    agent_t{{tile_base, agent_op}}.template consume_tile<true>(items_per_tile, threads_per_block);
  }} else {{
    agent_t{{tile_base, agent_op}}.template consume_tile<false>(items_in_tile, threads_per_block);
  }}
}}

)",
    mode_str);

  // Host wrapper function
  src += R"(extern "C" EXPORT int cccl_jit_binary_search(
    void* d_in_0, unsigned long long num_items,
    void* d_in_1, unsigned long long num_values,
    void* d_out_0, void* op_0_state
) {
)";
  src += "    " + data_code.setup_code + "\n";
  src += "    " + values_code.setup_code + "\n";
  src += "    " + out_code.setup_code + "\n";
  src += "    " + op_code.setup_code + "\n";
  src += R"(    if (num_values == 0) return 0;
    constexpr unsigned long long items_per_block = 512ULL;
    unsigned long long block_sz = (num_values + items_per_block - 1) / items_per_block;
    if (block_sz > (unsigned long long)UINT_MAX) return (int)cudaErrorInvalidValue;
    binary_search_kernel<<<(unsigned int)block_sz, 256>>>(in_0, num_items, in_1, num_values, out_0, op_0);
    return (int)cudaPeekAtLastError();
}
)";

  return src;
}

// Set up JITCompiler config — mirrors CubCall::compile() logic
static CompilerConfig make_binary_search_jit_config(
  int cc_major, int cc_minor, cccl_build_config* config, const char* ctk_root, const char* cccl_include_path)
{
  auto jit_config             = detectDefaultConfig();
  jit_config.sm_version       = cc_major * 10 + cc_minor;
  jit_config.verbose          = false;
  jit_config.entry_point_name = "cccl_jit_binary_search";

  if (ctk_root && ctk_root[0] != '\0')
  {
    jit_config.cuda_toolkit_path = ctk_root;
    jit_config.library_paths.clear();
    for (const char* subdir : {"lib64", "lib"})
    {
      auto candidate = std::filesystem::path(ctk_root) / subdir;
      if (std::filesystem::exists(candidate))
      {
        jit_config.library_paths.push_back(candidate.string());
      }
    }
  }
  if (cccl_include_path && cccl_include_path[0] != '\0')
  {
    jit_config.cccl_include_path = cccl_include_path;
    if (jit_config.hostjit_include_path.empty()
        || !std::filesystem::exists(jit_config.hostjit_include_path + "/hostjit/cuda_minimal"))
    {
      auto parent = std::filesystem::path(cccl_include_path).parent_path().string();
      if (std::filesystem::exists(parent + "/hostjit/cuda_minimal"))
      {
        jit_config.hostjit_include_path = parent;
      }
    }
  }
  if (config)
  {
    for (size_t i = 0; i < config->num_extra_include_dirs; ++i)
    {
      jit_config.include_paths.push_back(config->extra_include_dirs[i]);
    }
    for (size_t i = 0; i < config->num_extra_compile_flags; ++i)
    {
      std::string flag = config->extra_compile_flags[i];
      if (flag.substr(0, 2) == "-D")
      {
        auto eq = flag.find('=', 2);
        if (eq != std::string::npos)
        {
          jit_config.macro_definitions[flag.substr(2, eq - 2)] = flag.substr(eq + 1);
        }
        else
        {
          jit_config.macro_definitions[flag.substr(2)] = "";
        }
      }
    }
  }
  return jit_config;
}

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
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();

  auto jit_config = make_binary_search_jit_config(cc_major, cc_minor, config, ctk_root, cccl_include_path);
  cccl::detail::add_extra_cub_thrust_includes(jit_config, cub_path, thrust_path);

  // Collect bitcode from op and iterators
  uintptr_t unique_id = reinterpret_cast<uintptr_t>(build_ptr);
  BitcodeCollector bitcode(jit_config, unique_id);
  bitcode.add_op(op, "op_0");
  bitcode.add_iterator(d_data, "in_0");
  bitcode.add_iterator(d_values, "in_1");
  bitcode.add_iterator(d_out, "out_0");

  // Generate source
  std::string cuda_source = make_binary_search_source(d_data, d_values, d_out, op, mode);

  // Compile. unique_ptr owns the JITCompiler so any early throw frees it; we
  // .release() into build_ptr->jit_compiler (raw void*) on the success path.
  auto compiler = std::make_unique<JITCompiler>(jit_config);
  if (!compiler->compile(cuda_source))
  {
    std::string err = compiler->getLastError();
    bitcode.cleanup();
    throw std::runtime_error("binary_search compilation failed: " + err);
  }
  bitcode.cleanup();

  // Extract function pointer
  using fn_t = int (*)(void*, ...);
  auto fn    = compiler->getFunction<fn_t>("cccl_jit_binary_search");
  if (!fn)
  {
    throw std::runtime_error("binary_search function lookup failed: " + compiler->getLastError());
  }

  auto cubin = compiler->getCubin();

  build_ptr->cc         = cc_major * 10 + cc_minor;
  build_ptr->cubin      = nullptr;
  build_ptr->cubin_size = 0;
  if (!cubin.empty())
  {
    auto* cubin_copy = new char[cubin.size()];
    std::memcpy(cubin_copy, cubin.data(), cubin.size());
    build_ptr->cubin      = cubin_copy;
    build_ptr->cubin_size = cubin.size();
  }
  build_ptr->jit_compiler     = compiler.release();
  build_ptr->binary_search_fn = reinterpret_cast<void*>(fn);

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search_build(): %s\n", exc.what());
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
  CUstream /*stream*/)
{
  try
  {
    auto fn = reinterpret_cast<binary_search_fn_t>(build.binary_search_fn);
    if (!fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    int status = fn(d_data.state, num_items, d_values.state, num_values, d_out.state, op.state);
    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
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

  if (build_ptr->jit_compiler)
  {
    delete static_cast<JITCompiler*>(build_ptr->jit_compiler);
    build_ptr->jit_compiler = nullptr;
  }
  if (build_ptr->cubin)
  {
    delete[] static_cast<char*>(build_ptr->cubin);
    build_ptr->cubin = nullptr;
  }
  build_ptr->cubin_size       = 0;
  build_ptr->binary_search_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_search_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
