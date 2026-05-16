//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cccl/c/types.h>
#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

namespace cccl::detail
{
/**
 * @brief Extends a vector of compilation arguments with extra flags and include directories from a build config
 *
 * @param args The vector of arguments to extend
 * @param config The build configuration containing extra flags and include directories (can be nullptr)
 */
inline void extend_args_with_build_config(std::vector<const char*>& args, const cccl_build_config* config)
{
  if (config)
  {
    // Add extra compile flags
    for (size_t i = 0; i < config->num_extra_compile_flags; ++i)
    {
      args.push_back(config->extra_compile_flags[i]);
    }
    // Add include directories
    for (size_t i = 0; i < config->num_extra_include_dirs; ++i)
    {
      args.push_back("-I");
      args.push_back(config->extra_include_dirs[i]);
    }
  }
}

// Parse path arguments from the Python layer for use with hostjit.
// Returns the bare CCCL include path (strips "-I" prefix if present).
inline std::string parse_cccl_include_path(const char* libcudacxx_path)
{
  if (!libcudacxx_path || libcudacxx_path[0] == '\0')
  {
    return {};
  }
  std::string p = libcudacxx_path;
  if (p.substr(0, 2) == "-I")
  {
    p = p.substr(2);
  }
  return p;
}

// Returns the CTK root directory (strips "-I" prefix and "/include" suffix if present).
// On systems where the CUDA toolkit uses the `targets/<arch>/include` layout
// (e.g. /usr/local/cuda/targets/x86_64-linux/include), backs up to the real
// toolkit root so callers find `nvvm/libdevice/libdevice.10.bc`.
inline std::string parse_ctk_root(const char* ctk_path)
{
  if (!ctk_path || ctk_path[0] == '\0')
  {
    return {};
  }
  std::string p = ctk_path;
  if (p.substr(0, 2) == "-I")
  {
    p = p.substr(2);
  }
  std::filesystem::path fp(p);
  if (fp.filename() == "include")
  {
    fp = fp.parent_path();
  }
  if (fp.parent_path().filename() == "targets")
  {
    fp = fp.parent_path().parent_path();
  }
  return fp.string();
}

// In source-tree (dev) builds, cub/ and thrust/ live at sibling paths to
// libcudacxx/include rather than under a single CCCL_INCLUDE_PATH. The test
// harness passes them as `-I`-prefixed strings; hostjit's
// `internal-isystem` plumbing only honors a single `cccl_include_path` for
// libcudacxx/cub/thrust, so push the bare cub/thrust paths into
// `include_paths` (`-I <path>`) instead.
inline void
add_extra_cub_thrust_includes(hostjit::CompilerConfig& jit_config, const char* cub_path, const char* thrust_path)
{
  auto strip_dash_I = [](const char* in) -> std::string {
    if (!in || in[0] == '\0')
    {
      return {};
    }
    std::string p = in;
    if (p.size() >= 2 && p.substr(0, 2) == "-I")
    {
      p = p.substr(2);
    }
    return p;
  };
  auto add_if_dir = [&](const std::string& p) {
    if (!p.empty() && std::filesystem::exists(p))
    {
      jit_config.include_paths.push_back(p);
    }
  };
  add_if_dir(strip_dash_I(cub_path));
  add_if_dir(strip_dash_I(thrust_path));
}

// RAII helper for merging cub_path / thrust_path (`-I`-prefixed) into a
// `cccl_build_config*`'s `extra_include_dirs` before passing to
// `CubCall::compile()`. The merged config and the strings it points into are
// kept alive for the lifetime of this object.
//
// Usage:
//   MergedBuildConfig merged(build_config, cub_path, thrust_path);
//   ... .compile(cc_major, cc_minor, merged.get(), ctk_root, ccl_inc);
class MergedBuildConfig
{
public:
  MergedBuildConfig(const cccl_build_config* base, const char* cub_path, const char* thrust_path)
  {
    if (base)
    {
      merged_ = *base;
    }
    for (size_t i = 0; i < merged_.num_extra_include_dirs; ++i)
    {
      ptrs_.push_back(merged_.extra_include_dirs[i]);
    }
    auto add = [&](const char* p) {
      if (!p || p[0] == '\0')
      {
        return;
      }
      std::string s = p;
      if (s.size() >= 2 && s.substr(0, 2) == "-I")
      {
        s = s.substr(2);
      }
      owned_strs_.push_back(std::move(s));
    };
    add(cub_path);
    add(thrust_path);
    for (auto& s : owned_strs_)
    {
      ptrs_.push_back(s.c_str());
    }
    merged_.extra_include_dirs     = ptrs_.data();
    merged_.num_extra_include_dirs = ptrs_.size();
  }

  cccl_build_config* get()
  {
    return &merged_;
  }

private:
  cccl_build_config merged_{};
  std::vector<std::string> owned_strs_;
  std::vector<const char*> ptrs_;
};

// Build a CompilerConfig from the standard set of path parameters.
// Mirrors the configuration logic in CubCall::compile().
inline hostjit::CompilerConfig make_jit_config(
  int cc_major,
  int cc_minor,
  const char* ctk_root, // already parsed (bare CTK root)
  const char* cccl_include_path, // already parsed (bare CCCL include path)
  cccl_build_config* config,
  const char* entry_point_name = nullptr)
{
  auto jit_config       = hostjit::detectDefaultConfig();
  jit_config.sm_version = cc_major * 10 + cc_minor;
  jit_config.verbose    = false;
  if (entry_point_name)
  {
    jit_config.entry_point_name = entry_point_name;
  }
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
      if (flag.size() >= 2 && flag.substr(0, 2) == "-D")
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

// Build a JITCompiler from the standard set of path parameters.
inline std::unique_ptr<hostjit::JITCompiler> make_jit_compiler(
  int cc_major,
  int cc_minor,
  const char* ctk_root,
  const char* cccl_include_path,
  cccl_build_config* config,
  const char* entry_point_name = nullptr)
{
  return std::make_unique<hostjit::JITCompiler>(
    make_jit_config(cc_major, cc_minor, ctk_root, cccl_include_path, config, entry_point_name));
}

// Compile a CUDA source string and return (compiler, fn_ptr, cubin).
// The compiler is owned by the returned JITResult; transfer ownership to a
// raw `void*` build-result slot with `result.compiler.release()`.
struct JITResult
{
  std::unique_ptr<hostjit::JITCompiler> compiler;
  void* fn_ptr = nullptr;
  std::vector<char> cubin;
};

inline JITResult compile_jit_source(
  const std::string& source,
  const char* fn_name,
  int cc_major,
  int cc_minor,
  const char* ctk_root,
  const char* cccl_include_path,
  cccl_build_config* config)
{
  auto compiler = make_jit_compiler(cc_major, cc_minor, ctk_root, cccl_include_path, config, fn_name);
  if (!compiler->compile(source))
  {
    fprintf(stderr, "\nJIT compilation failed: %s\n", compiler->getLastError().c_str());
    return {};
  }
  void* fn_ptr = compiler->getFunction<void*>(fn_name);
  if (!fn_ptr)
  {
    fprintf(stderr, "\nJIT symbol lookup failed for '%s': %s\n", fn_name, compiler->getLastError().c_str());
    return {};
  }
  JITResult result;
  result.fn_ptr   = fn_ptr;
  result.cubin    = compiler->getCubin();
  result.compiler = std::move(compiler);
  return result;
}

// Copy cubin data into a heap-allocated buffer and store size; returns pointer (caller frees with delete[]).
inline void* copy_cubin(const std::vector<char>& cubin, size_t* out_size)
{
  if (cubin.empty())
  {
    if (out_size)
    {
      *out_size = 0;
    }
    return nullptr;
  }
  auto* buf = new char[cubin.size()];
  std::memcpy(buf, cubin.data(), cubin.size());
  if (out_size)
  {
    *out_size = cubin.size();
  }
  return buf;
}
} // namespace cccl::detail
