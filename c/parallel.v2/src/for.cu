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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>

#include <cccl/c/for.h>
#include <hostjit/codegen/bitcode.hpp>
#include <hostjit/codegen/types.hpp>
#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>
#include <util/build_utils.h>

using namespace hostjit;
using namespace hostjit::codegen;

// d_in_0, num_items, op_0_state
using for_fn_t = int (*)(void*, unsigned long long, void*);

static std::string make_for_source(cccl_iterator_t d_data, cccl_op_t op)
{
  const bool has_bc   = BitcodeCollector::is_bitcode_op(op);
  const bool stateful = (op.type == CCCL_STATEFUL);
  const std::string op_name(op.name ? op.name : "op");

  // Resolve the element type: a builtin C name (e.g. "int") for primitive
  // value_types, or an emitted storage struct alias (e.g. "for_value_t") for
  // custom user types. The storage struct's `preamble` must come before the
  // first use of `data_type` in the rest of the source.
  std::string storage_preamble;
  const std::string data_type = resolve_type(d_data.value_type, "for_value_t", storage_preamble);

  std::string src = R"(#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/agent/agent_for.cuh>
#include <climits>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

)";

  src += storage_preamble;

  // Define the iterator type — always a raw pointer for pointer inputs
  src += std::format("using in_0_it_t = {}*;\n\n", data_type);

  // User op forward declaration or inline source
  if (op.code_type == CCCL_OP_CPP_SOURCE && op.code && op.code_size > 0)
  {
    src += std::string(op.code, op.code_size);
    src += "\n";
  }
  else if (has_bc)
  {
    if (stateful)
    {
      src += std::format("extern \"C\" __device__ void {}(void* state, {}* input);\n\n", op_name, data_type);
    }
    else
    {
      src += std::format("extern \"C\" __device__ void {}({}* input);\n\n", op_name, data_type);
    }
  }

  // user_op_t functor
  if (stateful)
  {
    // State bytes are embedded by value, not via host pointer; the bytes
    // travel into device constant memory through the kernel-arg copy when
    // CUB launches the kernel. See operators.cpp:generate_binary_functor.
    const size_t state_size  = op.size > 0 ? op.size : 1;
    const size_t state_align = op.alignment > 0 ? op.alignment : 1;
    src += std::format(
      "struct user_op_t {{\n"
      "  alignas({0}) unsigned char state_bytes[{1}];\n"
      "  __device__ __forceinline__ void operator()({2}* input) const "
      "{{ {3}((void*)state_bytes, input); }}\n"
      "}};\n\n",
      state_align,
      state_size,
      data_type,
      op_name);
  }
  else
  {
    src += std::format(
      R"(struct user_op_t {{
  __device__ __forceinline__ void operator()({}* input) const {{ {}(input); }}
}};

)",
      data_type,
      op_name);
  }

  // Policy
  src += R"(using OffsetT = unsigned long long;
using policy_dim_t = cub::detail::for_each::policy_t<256, 2>;
struct device_for_policy {
  struct ActivePolicy {
    using for_policy_t = policy_dim_t;
  };
};

)";

  // Template kernel
  src += std::format(
    R"(template<typename DataIt, typename OpT>
_CCCL_KERNEL_ATTRIBUTES
__launch_bounds__(device_for_policy::ActivePolicy::for_policy_t::threads_per_block)
void for_kernel(DataIt d_data, OffsetT num_items, OpT user_op)
{{
  auto agent_op = [&user_op, &d_data](OffsetT idx) {{
    user_op(d_data + idx);
  }};
  using active_policy_t = device_for_policy::ActivePolicy::for_policy_t;
  using agent_t = cub::detail::for_each::agent_block_striped_t<active_policy_t, OffsetT, decltype(agent_op)>;
  constexpr auto threads_per_block  = active_policy_t::threads_per_block;
  constexpr auto items_per_tile = active_policy_t::items_per_thread * threads_per_block;
  const auto tile_base     = static_cast<OffsetT>(blockIdx.x) * items_per_tile;
  const auto num_remaining = num_items - tile_base;
  const auto items_in_tile = static_cast<OffsetT>(num_remaining < items_per_tile ? num_remaining : items_per_tile);
  if (items_in_tile == items_per_tile) {{
    agent_t{{tile_base, agent_op}}.template consume_tile<true>(items_per_tile, threads_per_block);
  }} else {{
    agent_t{{tile_base, agent_op}}.template consume_tile<false>(items_in_tile, threads_per_block);
  }}
}}

)");

  // Host wrapper
  src += R"(extern "C" EXPORT int cccl_jit_for(
    void* d_in_0, unsigned long long num_items, void* op_0_state
) {
    in_0_it_t in_0 = static_cast<in_0_it_t>(d_in_0);
)";
  if (stateful)
  {
    const size_t state_size = op.size > 0 ? op.size : 1;
    src += std::format("    user_op_t op_0; __builtin_memcpy(op_0.state_bytes, op_0_state, {});\n", state_size);
  }
  else
  {
    src += "    user_op_t op_0{};\n";
  }
  src += R"(    if (num_items == 0) return 0;
    constexpr unsigned long long items_per_block = 512ULL;
    unsigned long long block_sz = (num_items + items_per_block - 1) / items_per_block;
    if (block_sz > (unsigned long long)UINT_MAX) return (int)cudaErrorInvalidValue;
    for_kernel<<<(unsigned int)block_sz, 256>>>(in_0, num_items, op_0);
    return (int)cudaPeekAtLastError();
}
)";

  return src;
}

// Set up JITCompiler config — mirrors binary_search.cu logic
static CompilerConfig make_for_jit_config(
  int cc_major, int cc_minor, cccl_build_config* config, const char* ctk_root, const char* cccl_include_path)
{
  auto jit_config             = detectDefaultConfig();
  jit_config.sm_version       = cc_major * 10 + cc_minor;
  jit_config.verbose          = false;
  jit_config.entry_point_name = "cccl_jit_for";

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

CUresult cccl_device_for_build_ex(
  cccl_device_for_build_result_t* build_ptr,
  cccl_iterator_t d_data,
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
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  auto jit_config = make_for_jit_config(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  // Collect bitcode from op
  uintptr_t unique_id = reinterpret_cast<uintptr_t>(build_ptr);
  BitcodeCollector bitcode(jit_config, unique_id);
  bitcode.add_op(op, "op_0");

  // Generate source
  std::string cuda_source = make_for_source(d_data, op);
  if (const char* dump_path = std::getenv("FOR_DUMP_SOURCE"))
  {
    std::ofstream f(dump_path);
    f << cuda_source;
  }

  // Compile. unique_ptr owns the JITCompiler so any early throw frees it; we
  // .release() into build_ptr->jit_compiler (raw void*) on the success path.
  auto compiler = std::make_unique<JITCompiler>(jit_config);
  if (!compiler->compile(cuda_source))
  {
    std::string err = compiler->getLastError();
    bitcode.cleanup();
    throw std::runtime_error("for compilation failed: " + err);
  }
  bitcode.cleanup();

  // Extract function pointer
  using fn_t = int (*)(void*, ...);
  auto fn    = compiler->getFunction<fn_t>("cccl_jit_for");
  if (!fn)
  {
    throw std::runtime_error("for function lookup failed: " + compiler->getLastError());
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
  build_ptr->jit_compiler = compiler.release();
  build_ptr->for_fn       = reinterpret_cast<void*>(fn);

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_for_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_for(
  cccl_device_for_build_result_t build, cccl_iterator_t d_data, uint64_t num_items, cccl_op_t op, CUstream /*stream*/)
{
  try
  {
    auto fn = reinterpret_cast<for_fn_t>(build.for_fn);
    if (!fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    int status = fn(d_data.state, num_items, op.state);
    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_for(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_for_build(
  cccl_device_for_build_result_t* build,
  cccl_iterator_t d_data,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_for_build_ex(
    build, d_data, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}

CUresult cccl_device_for_cleanup(cccl_device_for_build_result_t* build_ptr)
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
  build_ptr->cubin_size = 0;
  build_ptr->for_fn     = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_for_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
