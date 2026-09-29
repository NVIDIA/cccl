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

// Returns the full path to libdevice.10.bc given a CTK root (as returned by
// parse_ctk_root()).
//
// CUDA 13.x pip wheels merge every CTK component into one shared
// nvidia/cu13/ tree, so `<ctk_root>/nvvm/libdevice/libdevice.10.bc` (the
// layout used by the flat `/usr/local/cuda` install and by
// `/usr/local/cuda/targets/<arch>/include`) resolves directly. CUDA 12.x pip
// wheels instead install each CTK component into its own isolated package
// directory — e.g. cudart headers/libs under nvidia/cuda_runtime/, but
// libdevice under the sibling nvidia/cuda_nvcc/nvvm/libdevice/ — so
// `ctk_root` (derived from the cudart include path) never contains it
// directly. Fall back to scanning `ctk_root`'s siblings for that layout.
inline std::string find_libdevice_bc(const std::string& ctk_root)
{
  if (ctk_root.empty())
  {
    return {};
  }
  std::filesystem::path default_path = std::filesystem::path(ctk_root) / "nvvm" / "libdevice" / "libdevice.10.bc";
  if (std::filesystem::exists(default_path))
  {
    return default_path.string();
  }
  std::filesystem::path parent = std::filesystem::path(ctk_root).parent_path();
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(parent, ec))
  {
    if (!entry.is_directory(ec))
    {
      continue;
    }
    std::filesystem::path candidate = entry.path() / "nvvm" / "libdevice" / "libdevice.10.bc";
    if (std::filesystem::exists(candidate))
    {
      return candidate.string();
    }
  }
  // Not found anywhere; return the default location so callers get the same
  // "file not found" error they would have gotten before this fallback.
  return default_path.string();
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

// Copy cubin data into a heap-allocated buffer the caller owns. Plain `new[]`
// — memcpy is noexcept so there's no exception path between the allocation
// and the assignment to out_cubin. The caller eventually frees via
// release_jit_artifacts() (or delete[] on out_cubin).
inline void copy_cubin(const std::vector<char>& cubin, void*& out_cubin, size_t& out_size)
{
  if (cubin.empty())
  {
    out_cubin = nullptr;
    out_size  = 0;
    return;
  }
  auto* buf = new char[cubin.size()];
  std::memcpy(buf, cubin.data(), cubin.size());
  out_cubin = buf;
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
  delete[] static_cast<char*>(build_ptr->payload);
  build_ptr->payload      = nullptr;
  build_ptr->payload_size = 0;
}
} // namespace cccl::detail
