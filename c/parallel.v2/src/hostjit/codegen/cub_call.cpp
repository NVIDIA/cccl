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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>

#include <hostjit/codegen/bitcode.hpp>
#include <hostjit/codegen/cub_call.hpp>
#include <hostjit/codegen/iterators.hpp>
#include <hostjit/codegen/operators.hpp>
#include <hostjit/codegen/types.hpp>

namespace hostjit::codegen
{
CubCall CubCall::from(const char* include_header)
{
  CubCall c;
  c.include_ = include_header;
  return c;
}

CubCall& CubCall::run(const char* cub_function)
{
  cub_function_ = cub_function;
  return *this;
}

CubCall& CubCall::name(const char* export_name)
{
  fn_name_ = export_name;
  return *this;
}

// Helper to find the accumulator type from the argument list.
// Priority: first cccl_value_t, then first input_t's value_type.
namespace
{
cccl_type_info find_accum_type(const std::vector<Arg>& args)
{
  // Highest priority: explicit override
  for (const auto& arg : args)
  {
    if (auto* fa = std::get_if<force_accum_type_t>(&arg))
    {
      return fa->type;
    }
  }
  // First: look for cccl_value_t (init value defines accum type)
  for (const auto& arg : args)
  {
    if (auto* val = std::get_if<cccl_value_t>(&arg))
    {
      return val->type;
    }
  }
  // Second: future_val_t carries explicit type info
  for (const auto& arg : args)
  {
    if (auto* fv = std::get_if<future_val_t>(&arg))
    {
      return fv->type;
    }
  }
  // Fallback: first input iterator's value_type
  for (const auto& arg : args)
  {
    if (auto* inp = std::get_if<input_t>(&arg))
    {
      return inp->it.value_type;
    }
  }
  // Last resort: first output iterator
  for (const auto& arg : args)
  {
    if (auto* outp = std::get_if<output_t>(&arg))
    {
      return outp->it.value_type;
    }
  }
  return cccl_type_info{sizeof(int), alignof(int), CCCL_INT32};
}
} // anonymous namespace

namespace
{
// Returns true if `args` contains an env_stream_t — used to decide whether the
// shared-includes block needs to pull in <cuda/std/__execution/env.h>.
bool needs_env_include(const std::vector<Arg>& args)
{
  for (const auto& arg : args)
  {
    if (std::holds_alternative<env_stream_t>(arg))
    {
      return true;
    }
  }
  return false;
}

// Emits the system #includes + the CUB header + EXPORT macro defn. Hoisted
// from source() so multi-function compiles can emit this once and wrap N
// function bodies in N namespaces below.
std::string shared_includes(const std::string& cub_include, bool needs_tuple, bool needs_env)
{
  std::string src = R"(#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/std/iterator>
#include <cuda/std/functional>
#include <cuda/functional>
)";
  if (needs_tuple)
  {
    src += "#include <cuda/std/tuple>\n";
  }
  if (needs_env)
  {
    // Use the narrow internal env.h header rather than <cuda/std/execution>
    // — the umbrella header pulls in pstl machinery that depends on <vector>
    // and exception types not available in the hostjit environment.
    src += "#include <cuda/std/__execution/env.h>\n";
    src += "#include <cuda/stream_ref>\n";
  }
  src += std::format("#include <{}>\n\n", cub_include);

  src += R"(#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

)";
  return src;
}
} // namespace

std::string CubCall::source() const
{
  // Single-function source = shared includes/EXPORT + this CubCall's body.
  return shared_includes(include_, tuple_inputs_, needs_env_include(args_)) + body();
}

std::string CubCall::body() const
{
  // Pass 1: determine accumulator type
  cccl_type_info accum_info = find_accum_type(args_);
  std::string accum_preamble;
  std::string accum_type = resolve_type(accum_info, "storage_t", accum_preamble);

  // Counters for unique naming
  int in_count  = 0;
  int out_count = 0;
  int op_count  = 0;
  int val_count = 0;

  // Accumulated sections
  std::string preamble;
  std::vector<std::string> params;
  std::vector<std::string> setup_lines;
  std::vector<std::string> cub_args;
  // Lines emitted after the cub::DeviceX::Y(...) call and before the return —
  // populated by post-call tags (e.g. selector_out_t capturing a DoubleBuffer's
  // selector member).
  std::vector<std::string> post_call_lines;

  // Emit accum type
  if (!accum_preamble.empty())
  {
    preamble += accum_preamble;
  }
  preamble += std::format("using accum_t = {};\n\n", accum_type);

  // Shared alias cache: (size, alignment) → type name.
  // Multiple iterators with the same unknown struct layout must share a single C++
  // type so that CUB can move data between them (e.g. merge sort block loads).
  std::map<std::pair<size_t, size_t>, std::string> struct_type_map;
  int struct_type_counter = 0;

  // Return a stable C++ element-type name for an iterator's value_type:
  //   - Known C type  → C++ keyword (e.g. "int", "float")
  //   - Struct matching accum_t → "accum_t"  (preserves operator compatibility)
  //   - Other struct  → shared alias for this (size, alignment) layout
  // Built-in C type sizes (CCCL_TYPE_ENUM → bytes). Used to detect a
  // mismatch where the caller reports a primitive `vt.type` but `vt.size`
  // says the element is wider — common when a custom struct happens to
  // share the primitive's tag. In that case fall through to a storage
  // struct so the iterator strides correctly.
  auto builtin_size = [](cccl_type_enum t) -> size_t {
    switch (t)
    {
      case CCCL_INT8:
      case CCCL_UINT8:
      case CCCL_BOOLEAN:
        return 1;
      case CCCL_INT16:
      case CCCL_UINT16:
      case CCCL_FLOAT16:
        return 2;
      case CCCL_INT32:
      case CCCL_UINT32:
      case CCCL_FLOAT32:
        return 4;
      case CCCL_INT64:
      case CCCL_UINT64:
      case CCCL_FLOAT64:
        return 8;
      default:
        return 0;
    }
  };
  auto iter_elem_type_name = [&](const cccl_type_info& vt) -> std::string {
    auto name = get_type_name(vt.type);
    if (!name.empty() && vt.size == builtin_size(vt.type))
    {
      return name;
    }
    if (vt.size == accum_info.size && vt.alignment == accum_info.alignment && vt.type == accum_info.type)
    {
      return "accum_t";
    }
    auto key = std::make_pair(vt.size, vt.alignment);
    auto it  = struct_type_map.find(key);
    if (it != struct_type_map.end())
    {
      return it->second;
    }
    auto alias = std::format("__cccl_struct_{}_t", struct_type_counter++);
    preamble += make_storage_type(alias.c_str(), vt.size, vt.alignment);
    struct_type_map[key] = alias;
    return alias;
  };

  // Pass 2: process each argument
  for (const auto& arg : args_)
  {
    std::visit(
      [&](auto&& a) {
        using T = std::decay_t<decltype(a)>;

        if constexpr (std::is_same_v<T, temp_storage_t>)
        {
          params.push_back("void* d_temp_storage");
          cub_args.push_back("d_temp_storage");
        }
        else if constexpr (std::is_same_v<T, temp_bytes_t>)
        {
          params.push_back("size_t* temp_storage_bytes");
          cub_args.push_back("*temp_storage_bytes");
        }
        else if constexpr (std::is_same_v<T, num_items_t>)
        {
          params.push_back(std::format("unsigned long long {}", a.name));
          cub_args.push_back(std::format("(unsigned long long){}", a.name));
        }
        else if constexpr (std::is_same_v<T, stream_t>)
        {
          params.push_back("void* stream");
          cub_args.push_back("(cudaStream_t)stream");
        }
        else if constexpr (std::is_same_v<T, env_stream_t>)
        {
          params.push_back("void* stream");
          cub_args.push_back("::cuda::std::execution::env{::cuda::stream_ref{(cudaStream_t)stream}}");
        }
        else if constexpr (std::is_same_v<T, input_t>)
        {
          auto idx         = in_count++;
          auto struct_name = std::format("in_{}_it_t", idx);
          auto var_name    = std::format("in_{}", idx);
          auto param_name  = std::format("d_in_{}", idx);

          auto value_type = iter_elem_type_name(a.it.value_type);
          auto code       = make_input_iterator(a.it, value_type, "accum_t", struct_name, var_name, param_name);

          preamble += code.preamble;
          params.push_back(std::format("void* {}", param_name));
          setup_lines.push_back(code.setup_code);
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, output_t>)
        {
          auto idx         = out_count++;
          auto struct_name = std::format("out_{}_it_t", idx);
          auto var_name    = std::format("out_{}", idx);
          auto param_name  = std::format("d_out_{}", idx);

          auto value_type = iter_elem_type_name(a.it.value_type);
          auto code       = make_output_iterator(a.it, "accum_t", struct_name, var_name, param_name, value_type);

          preamble += code.preamble;
          params.push_back(std::format("void* {}", param_name));
          setup_lines.push_back(code.setup_code);
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, cccl_op_t>)
        {
          auto idx          = op_count++;
          auto functor_name = std::format("Op_{}", idx);
          auto var_name     = std::format("op_{}", idx);
          auto state_param  = std::format("op_{}_state", idx);
          bool has_bc       = BitcodeCollector::is_bitcode_op(a);

          auto code = make_binary_op(a, accum_type, functor_name, var_name, state_param, has_bc);

          preamble += code.preamble;
          // Always emit op_state param for ABI stability (unused for stateless ops)
          params.push_back(std::format("void* {}", state_param));
          setup_lines.push_back(code.setup_code);
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, cmp_t>)
        {
          auto idx          = op_count++;
          auto functor_name = std::format("CmpOp_{}", idx);
          auto var_name     = std::format("cmp_{}", idx);
          auto state_param  = std::format("cmp_{}_state", idx);
          bool has_bc       = BitcodeCollector::is_bitcode_op(a.op);

          auto code = make_comparison_op(a.op, accum_type, functor_name, var_name, state_param, has_bc);

          preamble += code.preamble;
          params.push_back(std::format("void* {}", state_param));
          setup_lines.push_back(code.setup_code);
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, for_each_op_t>)
        {
          auto idx          = op_count++;
          auto functor_name = std::format("ForEachOp_{}", idx);
          auto var_name     = std::format("op_{}", idx);
          auto state_param  = std::format("op_{}_state", idx);
          bool has_bc       = BitcodeCollector::is_bitcode_op(a.op);

          // The element type is the first input iterator's value_type, which
          // CubCall has already resolved via find_accum_type.
          const std::string elem_type = iter_elem_type_name(accum_info);

          auto code = make_for_each_op(a.op, elem_type, functor_name, var_name, state_param, has_bc);

          preamble += code.preamble;
          params.push_back(std::format("void* {}", state_param));
          setup_lines.push_back(code.setup_code);
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, double_buffer_t>)
        {
          // Emit two void* params (in/out buffer state pointers), construct a
          // cub::DoubleBuffer<elem_t> local with the given var_name, and pass
          // the buffer to the CUB call. iter_elem_type_name resolves the
          // element type the same way input/output iterators do.
          const std::string elem_type = iter_elem_type_name(a.in_it.value_type);
          const std::string var_name  = a.var_name;
          const auto in_param         = var_name + "_in_state";
          const auto out_param        = var_name + "_out_state";

          params.push_back(std::format("void* {}", in_param));
          params.push_back(std::format("void* {}", out_param));
          setup_lines.push_back(std::format(
            "cub::DoubleBuffer<{0}> {1}(static_cast<{0}*>({2}), static_cast<{0}*>({3}));",
            elem_type,
            var_name,
            in_param,
            out_param));
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, selector_out_t>)
        {
          // Emit a void* selector_out param and capture <buffer>.selector after
          // the CUB call. Paired with a double_buffer_t whose var_name matches.
          params.push_back("void* selector_out");
          post_call_lines.push_back(std::format("*static_cast<int*>(selector_out) = {}.selector;", a.buffer_var_name));
        }
        else if constexpr (std::is_same_v<T, unary_op_t>)
        {
          auto idx          = op_count++;
          auto functor_name = std::format("UnaryOp_{}", idx);
          auto var_name     = std::format("op_{}", idx);
          auto state_param  = std::format("op_{}_state", idx);
          bool has_bc       = BitcodeCollector::is_bitcode_op(a.op);

          // For unknown types the iterators use accum_t as fallback; the unary
          // op functor must use the same names so CUB can match the types.
          // Reuse the iterator's element-type resolver so a primitive `vt.type`
          // with a custom-sized `vt.size` falls back to the same storage alias
          // the iterator uses, rather than naming the wider element "int".
          std::string in_type  = iter_elem_type_name(a.in_type);
          std::string out_type = iter_elem_type_name(a.out_type);

          auto code = make_unary_op(a.op, in_type, out_type, functor_name, var_name, state_param, has_bc);

          preamble += code.preamble;
          params.push_back(std::format("void* {}", state_param));
          setup_lines.push_back(code.setup_code);
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, force_accum_type_t>)
        {
          // No-op: only influences accum type resolution, generates no code.
        }
        else if constexpr (std::is_same_v<T, future_val_t>)
        {
          auto idx        = val_count++;
          auto var_name   = std::format("future_{}", idx);
          auto param_name = std::format("future_{}_param", idx);

          // The caller passes a device pointer; we wrap it in FutureValue<accum_t>
          // so CUB fetches the init value from device memory at scan time.
          params.push_back(std::format("void* {}", param_name));
          setup_lines.push_back(
            std::format("cub::FutureValue<accum_t> {}(static_cast<accum_t*>({}));", var_name, param_name));
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, cccl_value_t>)
        {
          auto idx        = val_count++;
          auto var_name   = std::format("val_{}", idx);
          auto param_name = std::format("val_{}_ptr", idx);

          params.push_back(std::format("void* {}", param_name));
          setup_lines.push_back(std::format(
            "accum_t {};\n    __builtin_memcpy(&{}, {}, sizeof(accum_t));", var_name, var_name, param_name));
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, typed_scalar_t>)
        {
          // Caller passes a host pointer; we memcpy onto the stack as the
          // requested C++ type before calling CUB. The C++ type comes from
          // a.type (cccl_type_info), the wrapper parameter is named
          // `<a.name>_ptr`, and the local that CUB sees is `<a.name>`.
          const std::string cpp_type = resolve_type(a.type, a.name, preamble);
          const auto param_name      = std::string(a.name) + "_ptr";
          params.push_back(std::format("void* {}", param_name));
          setup_lines.push_back(
            std::format("{0} {1};\n    __builtin_memcpy(&{1}, {2}, sizeof({0}));", cpp_type, a.name, param_name));
          cub_args.push_back(a.name);
        }
      },
      arg);
  }

  // When tuple_inputs_ is set, replace the individual input cub_args with a
  // single make_tuple(...) expression covering all of them.
  if (tuple_inputs_ && in_count > 1)
  {
    // Collect the first in_count cub_args that correspond to input iterators.
    // Inputs are emitted first among iterator args, so they occupy the leading
    // cub_args entries (after temp_storage/temp_bytes if present).
    // Reconstruct: find and replace the in_0..in_N-1 vars with make_tuple.
    std::vector<std::string> input_vars;
    std::vector<std::string> other_args;
    for (const auto& a : cub_args)
    {
      // Input vars are named "in_0", "in_1", etc.
      if (a.starts_with("in_") && a.size() >= 4 && std::isdigit(a[3]))
      {
        input_vars.push_back(a);
      }
      else
      {
        other_args.push_back(a);
      }
    }
    std::string tuple_arg = "::cuda::std::make_tuple(";
    for (size_t i = 0; i < input_vars.size(); ++i)
    {
      if (i)
      {
        tuple_arg += ", ";
      }
      tuple_arg += input_vars[i];
    }
    tuple_arg += ")";
    // Rebuild cub_args: replace all in_* with the single tuple arg (at original position of in_0)
    cub_args.clear();
    cub_args.push_back(tuple_arg);
    for (const auto& a : other_args)
    {
      cub_args.push_back(a);
    }
  }

  // Assemble the per-function body: preamble + extern "C" function defn.
  // System #includes and the EXPORT macro live in shared_includes(), emitted
  // once at TU scope by either source() (single-fn) or compile() (multi-fn).
  std::string src = preamble;

  // Function signature
  src += std::format("extern \"C\" EXPORT int {}(\n", fn_name_);
  for (size_t i = 0; i < params.size(); ++i)
  {
    src += "    " + params[i];
    if (i + 1 < params.size())
    {
      src += ",\n";
    }
  }
  src += ")\n{\n";

  // Setup code
  for (const auto& line : setup_lines)
  {
    src += "    " + line + "\n";
  }
  src += "\n";

  // CUB call
  src += std::format("    cudaError_t err = {}(\n", cub_function_);
  for (size_t i = 0; i < cub_args.size(); ++i)
  {
    src += "        " + cub_args[i];
    if (i + 1 < cub_args.size())
    {
      src += ",\n";
    }
  }
  src += ");\n\n";

  // Post-call lines (e.g., capturing a DoubleBuffer's selector).
  for (const auto& line : post_call_lines)
  {
    src += "    " + line + "\n";
  }
  if (!post_call_lines.empty())
  {
    src += "\n";
  }

  // Error return
  src += R"(    return (int)err;
}
)";

  return src;
}

hostjit::CompilerConfig CubCall::make_jit_config(
  int cc_major,
  int cc_minor,
  cccl_build_config* config,
  const char* ctk_path,
  const char* cccl_include_path,
  const std::string& entry_point_name)
{
  auto jit_config             = hostjit::detectDefaultConfig();
  jit_config.sm_version       = cc_major * 10 + cc_minor;
  jit_config.verbose          = false;
  jit_config.entry_point_name = entry_point_name;

  if (ctk_path && ctk_path[0] != '\0')
  {
    jit_config.cuda_toolkit_path = ctk_path;
    // Rebuild library_paths from the new toolkit root so the linker
    // can find libcudart.so in the pip-installed layout.
    jit_config.library_paths.clear();
    for (const char* subdir : {"lib64", "lib"})
    {
      auto candidate = std::filesystem::path(ctk_path) / subdir;
      if (std::filesystem::exists(candidate))
      {
        jit_config.library_paths.push_back(candidate.string());
      }
    }
  }
  if (cccl_include_path && cccl_include_path[0] != '\0')
  {
    jit_config.cccl_include_path = cccl_include_path;
    // When CCCL headers are pip-installed, the hostjit cuda_minimal headers
    // are installed alongside them under the parent directory:
    //   cccl_include_path = .../cuda/cccl/headers/include/
    //   hostjit headers  = .../cuda/cccl/headers/hostjit/cuda_minimal/
    // So derive hostjit_include_path as the parent of cccl_include_path.
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
      std::string_view flag = config->extra_compile_flags[i];
      if (flag.starts_with("-D"))
      {
        flag.remove_prefix(2);
        if (auto eq = flag.find('='); eq != std::string_view::npos)
        {
          jit_config.macro_definitions[std::string{flag.substr(0, eq)}] = std::string{flag.substr(eq + 1)};
        }
        else
        {
          jit_config.macro_definitions[std::string{flag}] = "";
        }
      }
    }
    jit_config.enable_pch = config->enable_pch != 0;
    jit_config.verbose    = config->verbose != 0;
  }

  return jit_config;
}

CubCallResult CubCall::compile(
  int cc_major, int cc_minor, cccl_build_config* config, const char* ctk_path, const char* cccl_include_path) const
{
  // 1. Configure compiler
  auto jit_config = make_jit_config(cc_major, cc_minor, config, ctk_path, cccl_include_path, fn_name_);

  // 2. Auto-collect bitcode from ops and iterators
  uintptr_t unique_id = reinterpret_cast<uintptr_t>(this);
  BitcodeCollector bitcode(jit_config, unique_id);

  int op_idx  = 0;
  int in_idx  = 0;
  int out_idx = 0;
  collect_bitcode(bitcode, op_idx, in_idx, out_idx);

  // 3. Generate source
  std::string cuda_source = source();
  if (const char* dump_path = std::getenv("CUBCALL_DUMP_SOURCE"))
  {
    std::ofstream f(dump_path);
    f << cuda_source;
  }

  // 4. Compile. unique_ptr ensures the JITCompiler is freed if the next two
  // checks throw; .release() transfers ownership to CubCallResult on success.
  auto compiler = std::make_unique<JITCompiler>(jit_config);
  if (!compiler->compile(cuda_source))
  {
    std::string err = compiler->getLastError();
    bitcode.cleanup();
    throw std::runtime_error("CubCall compilation failed: " + err);
  }

  bitcode.cleanup();

  // 5. Extract function pointer
  using fn_t = int (*)(void*, ...);
  auto fn    = compiler->getFunction<fn_t>(fn_name_);
  if (!fn)
  {
    throw std::runtime_error("CubCall function lookup failed: " + compiler->getLastError());
  }

  // 6. Copy cubin
  auto cubin = compiler->getCubin();

  return CubCallResult{compiler.release(), reinterpret_cast<void*>(fn), std::move(cubin)};
}

void CubCall::collect_bitcode(BitcodeCollector& bitcode, int& op_idx, int& in_idx, int& out_idx) const
{
  for (const auto& arg : args_)
  {
    std::visit(
      [&](auto&& a) {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, cccl_op_t>)
        {
          bitcode.add_op(a, std::format("op_{}", op_idx++));
        }
        else if constexpr (std::is_same_v<T, cmp_t>)
        {
          bitcode.add_op(a.op, std::format("cmp_{}", op_idx++));
        }
        else if constexpr (std::is_same_v<T, unary_op_t>)
        {
          bitcode.add_op(a.op, std::format("op_{}", op_idx++));
        }
        else if constexpr (std::is_same_v<T, for_each_op_t>)
        {
          bitcode.add_op(a.op, std::format("op_{}", op_idx++));
        }
        else if constexpr (std::is_same_v<T, input_t>)
        {
          bitcode.add_iterator(a.it, std::format("in_{}", in_idx++));
        }
        else if constexpr (std::is_same_v<T, output_t>)
        {
          bitcode.add_iterator(a.it, std::format("out_{}", out_idx++));
        }
      },
      arg);
  }
}

MultiCubCallResult CubCall::compile(
  std::initializer_list<CubCall> calls,
  int cc_major,
  int cc_minor,
  cccl_build_config* config,
  const char* ctk_path,
  const char* cccl_include_path)
{
  if (calls.size() == 0)
  {
    throw std::runtime_error("CubCall::compile: empty CubCall list");
  }

  // All CubCalls must share the same CUB header — we emit it once at the top
  // of the merged TU. (If a future use case needs heterogeneous includes,
  // extend this to union the set; for now keep it strict so silent mismatches
  // can't slip through.)
  const std::string& shared_include = calls.begin()->include_;
  for (const auto& cb : calls)
  {
    if (cb.include_ != shared_include)
    {
      throw std::runtime_error("CubCall::compile: all CubCalls in a multi-compile must share the same .from(include) "
                               "header");
    }
  }

  // Detect whether any CubCall needs the env / tuple system includes.
  bool any_tuple = false;
  bool any_env   = false;
  for (const auto& cb : calls)
  {
    any_tuple = any_tuple || cb.tuple_inputs_;
    any_env   = any_env || needs_env_include(cb.args_);
  }

  // entry_point_name is used to mark a single function as preserved during
  // internalization. Use the first CubCall's name as the primary entry; the
  // others will still be exported via extern "C" EXPORT so dlsym finds them.
  auto jit_config = make_jit_config(cc_major, cc_minor, config, ctk_path, cccl_include_path, calls.begin()->fn_name_);

  // Shared BitcodeCollector across all CubCalls — identical user-op or
  // iterator bitcode referenced from multiple wrappers gets deduplicated by
  // content hash + symbol name inside the collector.
  uintptr_t unique_id = reinterpret_cast<uintptr_t>(&*calls.begin());
  BitcodeCollector bitcode(jit_config, unique_id);

  int op_idx  = 0;
  int in_idx  = 0;
  int out_idx = 0;
  for (const auto& cb : calls)
  {
    cb.collect_bitcode(bitcode, op_idx, in_idx, out_idx);
  }

  // Build the merged source: shared includes + EXPORT macro at TU scope,
  // then one `namespace fn_<i> { ... body() }` per CubCall. The extern "C"
  // EXPORT symbols defined inside each namespace export under the global
  // C-linkage name (no mangling), so dlsym(handle, cb.fn_name_) finds them.
  std::string cuda_source = shared_includes(shared_include, any_tuple, any_env);
  int i                   = 0;
  for (const auto& cb : calls)
  {
    cuda_source += std::format("namespace fn_{} {{\n", i);
    cuda_source += cb.body();
    cuda_source += std::format("}} // namespace fn_{}\n\n", i);
    ++i;
  }

  if (const char* dump_path = std::getenv("CUBCALL_DUMP_SOURCE"))
  {
    std::ofstream f(dump_path);
    f << cuda_source;
  }
  if (std::getenv("CUBCALL_PRINT_SOURCE"))
  {
    std::fprintf(stderr,
                 "\n===== CubCall merged JIT source [%zu fns] =====\n%s\n===== end =====\n",
                 calls.size(),
                 cuda_source.c_str());
  }

  // Single Clang compile for the whole TU.
  auto compiler = std::make_unique<JITCompiler>(jit_config);
  if (!compiler->compile(cuda_source))
  {
    std::string err = compiler->getLastError();
    bitcode.cleanup();
    throw std::runtime_error("CubCall::compile (multi) compilation failed: " + err);
  }
  bitcode.cleanup();

  // dlsym each function by its export name (positional order matches input).
  using fn_t = int (*)(void*, ...);
  std::vector<void*> fn_ptrs;
  fn_ptrs.reserve(calls.size());
  for (const auto& cb : calls)
  {
    auto fn = compiler->getFunction<fn_t>(cb.fn_name_);
    if (!fn)
    {
      throw std::runtime_error("CubCall::compile (multi) function lookup failed: " + cb.fn_name_);
    }
    fn_ptrs.push_back(reinterpret_cast<void*>(fn));
  }

  auto cubin = compiler->getCubin();
  return MultiCubCallResult{compiler.release(), std::move(cubin), std::move(fn_ptrs)};
}
} // namespace hostjit::codegen
