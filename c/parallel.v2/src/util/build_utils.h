//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
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

// Copy cubin data into a heap-allocated buffer the caller owns. std::make_unique
// gives RAII between alloc and assign so a throw on the memcpy can't leak.
// Caller eventually frees via release_jit_artifacts() (or delete[] on out_cubin).
inline void copy_cubin(const std::vector<char>& cubin, void*& out_cubin, size_t& out_size)
{
  if (cubin.empty())
  {
    out_cubin = nullptr;
    out_size  = 0;
    return;
  }
  auto buf = std::make_unique<char[]>(cubin.size());
  std::memcpy(buf.get(), cubin.data(), cubin.size());
  out_cubin = buf.release();
  out_size  = cubin.size();
}

// Free the JIT compiler and cubin buffer common to every build_result_t in
// c/parallel.v2/. Algorithm-specific fields (X_fn, determinism, etc.) get
// nulled by the caller after this. Template'd over the build_result type so
// each algorithm header doesn't need to include this one transitively.
template <typename BuildResult>
void release_jit_artifacts(BuildResult* build_ptr)
{
  delete static_cast<hostjit::JITCompiler*>(build_ptr->jit_compiler);
  build_ptr->jit_compiler = nullptr;
  delete[] static_cast<char*>(build_ptr->cubin);
  build_ptr->cubin      = nullptr;
  build_ptr->cubin_size = 0;
}
} // namespace cccl::detail
