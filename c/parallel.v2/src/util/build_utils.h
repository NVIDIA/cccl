//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <cccl/c/types.h>
#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

namespace cccl::detail
{
// Strip a leading "-I" prefix from `s`. Centralizes the prefix-handling used
// by every entry point that accepts an `-I`-prefixed include path from the
// Python layer (which speaks compile-flag syntax) and converts it to a bare
// filesystem path (which hostjit's CompilerConfig wants).
//
// Returns an empty string for null / empty input — every caller treats that
// as "no path supplied" so the bare check stays in one place.
inline std::string strip_dash_i_prefix(const char* s)
{
  if (!s || s[0] == '\0')
  {
    return {};
  }
  std::string_view sv{s};
  if (sv.starts_with("-I"))
  {
    sv.remove_prefix(2);
  }
  return std::string{sv};
}

// Parse path arguments from the Python layer for use with hostjit.
// Returns the bare CCCL include path (strips "-I" prefix if present).
inline std::string parse_cccl_include_path(const char* libcudacxx_path)
{
  return strip_dash_i_prefix(libcudacxx_path);
}

// Returns the CTK root directory (strips "-I" prefix and "/include" suffix if
// present, then walks up to the directory that contains `nvvm/libdevice/`).
// Works for both the flat `/usr/local/cuda` layout and the
// `/usr/local/cuda/targets/<arch>/include` layout (and any other arrangement)
// because it locates the toolkit root by its `nvvm/libdevice/libdevice.10.bc`
// marker rather than by hard-coded directory structure.
inline std::string parse_ctk_root(const char* ctk_path)
{
  std::string p = strip_dash_i_prefix(ctk_path);
  if (p.empty())
  {
    return {};
  }
  std::filesystem::path fp(p);
  if (fp.filename() == "include")
  {
    fp = fp.parent_path();
  }
  for (auto candidate = fp; candidate.has_parent_path() && candidate != candidate.parent_path();
       candidate      = candidate.parent_path())
  {
    if (std::filesystem::exists(candidate / "nvvm" / "libdevice"))
    {
      return candidate.string();
    }
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
  auto add_if_dir = [&](const std::string& p) {
    if (!p.empty() && std::filesystem::exists(p))
    {
      jit_config.include_paths.push_back(p);
    }
  };
  add_if_dir(strip_dash_i_prefix(cub_path));
  add_if_dir(strip_dash_i_prefix(thrust_path));
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
    // We append at most two paths (cub + thrust). Reserve up front so the
    // owned_strs_/ptrs_ vectors don't reallocate — important because we
    // capture pointers into owned_strs_ for `extra_include_dirs`.
    owned_strs_.reserve(2);
    ptrs_.reserve(merged_.num_extra_include_dirs + 2);

    for (size_t i = 0; i < merged_.num_extra_include_dirs; ++i)
    {
      ptrs_.push_back(merged_.extra_include_dirs[i]);
    }
    auto add = [&](const char* p) {
      auto s = strip_dash_i_prefix(p);
      if (!s.empty())
      {
        owned_strs_.push_back(std::move(s));
      }
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

// Copy byte data into a heap-allocated buffer the caller owns. Plain `new[]`
// — memcpy is noexcept so there's no exception path between the allocation
// and the assignment to out_bytes. The caller eventually frees via
// release_jit_artifacts() (or delete[] on out_bytes). Used to populate
// build_result_t::payload, which holds the full compiled .so/.dll bytes
// (not just the raw device cubin — the host wrapper code needs to travel
// with it too, since a v2 build_result is a native shared library, not a
// driver-loadable device blob).
inline void copy_bytes(const std::vector<char>& bytes, void*& out_bytes, size_t& out_size)
{
  if (bytes.empty())
  {
    out_bytes = nullptr;
    out_size  = 0;
    return;
  }
  auto* buf = new char[bytes.size()];
  std::memcpy(buf, bytes.data(), bytes.size());
  out_bytes = buf;
  out_size  = bytes.size();
}

// Build a hostjit::CompilerConfig suitable for JITCompiler::loadFromBytes(),
// honoring an optional caller-supplied ctk_path override (NULL = auto-detect
// via detectDefaultConfig()). This is the load-time counterpart of
// CubCall's own ctk_path handling at build time — needed because a
// deserialized/persisted artifact's baked RPATH may point at a different
// (build) machine's CUDA Toolkit; the loader must locate ITS OWN CUDA
// Toolkit to preload libcudart before dlopen (see
// hostjit::JITCompiler::loadFromBytes for why).
inline hostjit::CompilerConfig make_load_jit_config(const char* ctk_path)
{
  hostjit::CompilerConfig config = hostjit::detectDefaultConfig();
  std::string ctk_root_str       = parse_ctk_root(ctk_path);
  if (!ctk_root_str.empty())
  {
    config.cuda_toolkit_path = ctk_root_str;
    config.library_paths.clear();
    for (const char* subdir : {"lib64", "lib"})
    {
      auto candidate = std::filesystem::path(ctk_root_str) / subdir;
      if (std::filesystem::exists(candidate))
      {
        config.library_paths.push_back(candidate.string());
      }
    }
  }
  return config;
}

// Read the compiled shared library bytes for a live (already compile()'d)
// JITCompiler. Used by *_build_ex (the fused compile+load path) to ALSO
// populate build->payload with the AoT-serializable artifact, so
// *_serialize() works uniformly regardless of whether a build_result came
// from _build_ex (fused) or _compile()+_load() (split). Returns an empty
// vector (not an error) if the artifact can't be read — callers treat a
// resulting null/empty payload as "not serializable", not a build failure.
inline std::vector<char> read_compiled_library_bytes(hostjit::JITCompiler* compiler)
{
#ifdef _WIN32
  std::string lib_path = compiler->getArtifactsPath() + "\\cuda_code.dll";
#else
  std::string lib_path = compiler->getArtifactsPath() + "/libcuda_code.so";
#endif
  std::ifstream lib_file(lib_path, std::ios::binary);
  if (!lib_file)
  {
    return {};
  }
  return std::vector<char>((std::istreambuf_iterator<char>(lib_file)), std::istreambuf_iterator<char>());
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
  delete[] static_cast<char*>(build_ptr->payload);
  build_ptr->payload      = nullptr;
  build_ptr->payload_size = 0;
}
} // namespace cccl::detail
