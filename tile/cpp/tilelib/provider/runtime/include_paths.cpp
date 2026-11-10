// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif
#include <nvrtc.h>

#include "provider/runtime/runtime.h"

#if defined(_WIN32)
// NVRTC's declarations omit dllimport. Use the loader-populated import pointer
// so module lookup sees NVRTC itself rather than this provider's import thunk.
extern "C" decltype(&nvrtcVersion) __imp_nvrtcVersion;
#endif

namespace cuda::coop::tile::detail::provider
{
namespace
{
namespace fs = std::filesystem;

inline constexpr char encoded_include_path_separator = '\x1f';
#if defined(_WIN32)
inline constexpr char native_path_list_separator = ';';
#else
inline constexpr char native_path_list_separator = ':';
#endif

template <typename Address>
[[nodiscard]] fs::path module_path_for_address(Address address)
{
#if defined(_WIN32)
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(address),
                         &module)
      == 0)
  {
    return fs::path();
  }

  std::vector<wchar_t> path(512);
  while (path.size() <= 32768)
  {
    const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0)
    {
      return fs::path();
    }
    if (length < path.size())
    {
      return fs::path(std::wstring(path.data(), length));
    }
    path.resize(path.size() * 2);
  }
  return fs::path();
#else
  Dl_info module_info{};
  if (dladdr(reinterpret_cast<void*>(address), &module_info) != 0 && module_info.dli_fname
      && module_info.dli_fname[0] != '\0')
  {
    return fs::path(module_info.dli_fname);
  }
  return fs::path();
#endif
}

[[nodiscard]] fs::path nvrtc_module_path()
{
#if defined(_WIN32)
  return module_path_for_address(__imp_nvrtcVersion);
#else
  return module_path_for_address(&nvrtcVersion);
#endif
}

[[nodiscard]] bool is_directory_path(const fs::path& path)
{
  std::error_code ec;
  return fs::is_directory(path, ec);
}

[[nodiscard]] bool is_file_path(const fs::path& path)
{
  std::error_code ec;
  return fs::is_regular_file(path, ec);
}

[[nodiscard]] bool is_cccl_root(const fs::path& path)
{
  return is_file_path(path / "cub" / "cub" / "version.cuh") && is_directory_path(path / "libcudacxx" / "include")
      && is_directory_path(path / "thrust");
}

void push_unique_existing_path(
  const fs::path& path, std::vector<std::string>* out, std::unordered_set<std::string>* seen)
{
  if (!out || !seen || path.empty())
  {
    return;
  }
  const fs::path normalized = path.lexically_normal();
  if (!is_directory_path(normalized))
  {
    return;
  }
  const std::string text = normalized.string();
  if (!text.empty() && seen->insert(text).second)
  {
    out->push_back(text);
  }
}

void append_cccl_include_paths(
  const fs::path& cccl_root, std::vector<std::string>* out, std::unordered_set<std::string>* seen)
{
  if (cccl_root.empty())
  {
    return;
  }
  push_unique_existing_path(cccl_root / "libcudacxx" / "include", out, seen);
  push_unique_existing_path(cccl_root / "cub", out, seen);
  push_unique_existing_path(cccl_root / "thrust", out, seen);
}

[[nodiscard]] fs::path header_bundle(const fs::path& root)
{
  for (const fs::path& candidate : {root, root / "include"})
  {
    if (is_file_path(candidate / "cub" / "version.cuh") && !is_file_path(candidate / "cuda_runtime.h"))
    {
      return candidate;
    }
  }
  return {};
}

[[nodiscard]] std::string configured_cccl_root(fs::path* selected)
{
  selected->clear();
  for (const char* variable : {"CUDA_COOP_CCCL_ROOT", "CCCL_ROOT", "CCCL_DIR"})
  {
    const char* const value = std::getenv(variable);
    if (!value || value[0] == '\0')
    {
      continue;
    }
    std::error_code ec;
    const fs::path candidate = fs::canonical(value, ec);
    if (ec || (!is_cccl_root(candidate) && header_bundle(candidate).empty()))
    {
      return std::string("[LP-HEADERS-001] ") + variable
           + " does not name a CCCL source checkout or header bundle: " + value;
    }
    *selected = candidate;
    return {};
  }
  return {};
}

void append_selected_cccl_paths(
  const fs::path& selected, std::vector<std::string>* out, std::unordered_set<std::string>* seen)
{
  if (is_cccl_root(selected))
  {
    append_cccl_include_paths(selected, out, seen);
  }
  else
  {
    push_unique_existing_path(header_bundle(selected), out, seen);
  }
}

[[nodiscard]] fs::path find_build_cccl_root(const fs::path& provider_dir)
{
  // The build-only sidecar supports binary directories outside the checkout.
  // It is never installed or embedded in the provider's distributable artifacts.
  if (!is_directory_path(provider_dir / "CMakeFiles" / "cuda_coop_tile_provider.dir"))
  {
    return {};
  }
  std::ifstream source_file(provider_dir / "cuda-coop-tile-cccl-root.txt");
  std::string source_root;
  if (std::getline(source_file, source_root) && is_cccl_root(source_root))
  {
    return fs::path(source_root);
  }
  return {};
}

[[nodiscard]] std::string configured_cuda_root(fs::path* selected)
{
  selected->clear();
  const char* selected_variable = nullptr;
  for (const char* variable : {"CUDA_COOP_TILE_CUDA_ROOT", "CUDA_HOME", "CUDA_ROOT", "CUDA_PATH", "CUDA_TOOLKIT_PATH"})
  {
    const char* const value = std::getenv(variable);
    if (!value || value[0] == '\0')
    {
      continue;
    }
    std::error_code ec;
    const fs::path candidate = fs::canonical(value, ec);
    if (ec || !is_directory_path(candidate))
    {
      return std::string("[LP-TOOLKIT-001] ") + variable + " does not name a CUDA toolkit directory: " + value;
    }
    if (!selected->empty() && !fs::equivalent(candidate, *selected, ec))
    {
      return std::string("[LP-TOOLKIT-001] conflicting CUDA toolkit roots: ") + selected_variable + "="
           + selected->string() + ", " + variable + "=" + candidate.string();
    }
    *selected         = candidate;
    selected_variable = variable;
  }
  return {};
}

void append_cuda_root_include_paths(
  const fs::path& root, std::vector<std::string>* out, std::unordered_set<std::string>* seen)
{
  for (const fs::path& relative :
       {fs::path("bin/nvrtc/include"), fs::path("nvrtc/include"), fs::path("include"), fs::path("bin/include")})
  {
    const fs::path candidate = root / relative;
    if (is_file_path(candidate / "nvrtc.h") || is_file_path(candidate / "cuda_runtime.h"))
    {
      push_unique_existing_path(candidate, out, seen);
    }
  }
}

void append_cuda_include_path(std::vector<std::string>* out, std::unordered_set<std::string>* seen)
{
  fs::path selected;
  if (!configured_cuda_root(&selected).empty())
  {
    return; // The provider callbacks report the configuration error before compilation.
  }
  if (!selected.empty())
  {
    append_cuda_root_include_paths(selected, out, seen);
    return;
  }

  const fs::path nvrtc_path = nvrtc_module_path();
  if (!nvrtc_path.empty())
  {
    const fs::path nvrtc_lib_dir = nvrtc_path.lexically_normal().parent_path();
    for (fs::path current = nvrtc_lib_dir; !current.empty(); current = current.parent_path())
    {
      append_cuda_root_include_paths(current, out, seen);
      const fs::path parent = current.parent_path();
      if (parent == current)
      {
        break;
      }
    }
  }

  const char* const cudacxx = std::getenv("CUDACXX");
  if (cudacxx && cudacxx[0] != '\0')
  {
    const fs::path cudacxx_path = fs::path(cudacxx).lexically_normal();
    append_cuda_root_include_paths(cudacxx_path.parent_path().parent_path(), out, seen);
  }

#if !defined(_WIN32)
  append_cuda_root_include_paths("/usr/local/cuda", out, seen);
#endif
}
} // namespace

// The compiler process has already loaded NVRTC through the provider's native
// dependency. Validate that selection; changing the loader here is too late.
// Compare file identity so a toolkit assembled from component symlinks works.
bool validate_provider_configuration(std::string* out_error)
{
  fs::path selected;
  *out_error = configured_cccl_root(&selected);
  if (!out_error->empty())
  {
    return false;
  }
  *out_error = configured_cuda_root(&selected);
  if (!out_error->empty() || selected.empty())
  {
    return out_error->empty();
  }

  std::vector<std::string> includes;
  std::unordered_set<std::string> seen;
  append_cuda_root_include_paths(selected, &includes, &seen);
  if (!include_paths_contain_file(includes, "cuda_runtime.h"))
  {
    *out_error = "[LP-TOOLKIT-001] selected CUDA toolkit has no cuda_runtime.h: " + selected.string();
    return false;
  }

  std::error_code loaded_error;
  const fs::path loaded = fs::canonical(nvrtc_module_path(), loaded_error);
  if (loaded_error || loaded.empty())
  {
    *out_error = "[LP-TOOLKIT-001] cannot determine the loaded NVRTC library for CUDA toolkit " + selected.string();
    return false;
  }
  for (const fs::path& relative :
       {fs::path("bin/nvrtc/lib64"),
        fs::path("nvrtc/lib64"),
        fs::path("lib64"),
        fs::path("lib"),
        fs::path("bin/x64"),
        fs::path("bin/x86_64"),
        fs::path("bin/arm64"),
        fs::path("bin")})
  {
    const fs::path directory = selected / relative;
    std::vector<fs::path> candidates{directory / loaded.filename()};
#if !defined(_WIN32)
    candidates.push_back(directory / "libnvrtc.so");
    const std::string filename = loaded.filename().string();
    const std::string prefix   = "libnvrtc.so.";
    if (filename.compare(0, prefix.size(), prefix) == 0)
    {
      candidates.push_back(directory / filename.substr(0, filename.find('.', prefix.size())));
    }
#endif
    for (const fs::path& candidate : candidates)
    {
      std::error_code ec;
      if (fs::equivalent(candidate, loaded, ec) && !ec)
      {
        emit_trace_line("[coop_linkage_provider][trace] toolkit=" + selected.string() + " nvrtc=" + loaded.string());
        return true;
      }
    }
  }
  *out_error =
    "[LP-TOOLKIT-001] loaded NVRTC " + loaded.string() + " does not belong to the selected CUDA toolkit "
    + selected.string() + "; select the toolkit's compiler library directory before starting Python or the C++ build";
  return false;
}

std::vector<std::string> split_path_list(const std::string& value)
{
  std::vector<std::string> parts;
  if (value.empty())
  {
    return parts;
  }
  size_t start = 0;
  while (start <= value.size())
  {
    size_t end = value.find(native_path_list_separator, start);
    if (end == std::string::npos)
    {
      end = value.size();
    }
    if (end > start)
    {
      parts.emplace_back(value.substr(start, end - start));
    }
    if (end == value.size())
    {
      break;
    }
    start = end + 1;
  }
  return parts;
}

std::vector<std::string> split_encoded_path_list(const std::string& value)
{
  if (value.find(encoded_include_path_separator) == std::string::npos)
  {
    return split_path_list(value);
  }

  std::vector<std::string> parts;
  size_t start = 0;
  while (start <= value.size())
  {
    size_t end = value.find(encoded_include_path_separator, start);
    if (end == std::string::npos)
    {
      end = value.size();
    }
    if (end > start)
    {
      parts.emplace_back(value.substr(start, end - start));
    }
    if (end == value.size())
    {
      break;
    }
    start = end + 1;
  }
  return parts;
}

// Explicit roots take precedence over the provider's installed header bundle.
// Build-tree providers may use their source checkout when no bundle is present.
std::vector<std::string> resolve_provider_include_paths()
{
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;

  fs::path cccl_root;
  if (!configured_cccl_root(&cccl_root).empty())
  {
    return out; // Provider callbacks report the invalid explicit input.
  }
  if (!cccl_root.empty())
  {
    append_selected_cccl_paths(cccl_root, &out, &seen);
  }
  else
  {
    const fs::path provider_path = module_path_for_address(&resolve_provider_include_paths);
    if (!provider_path.empty())
    {
      const fs::path provider_dir = provider_path.lexically_normal().parent_path();
#if defined(CUDA_COOP_TILE_INSTALLED_INCLUDE_RELATIVE)
      const fs::path installed_headers = header_bundle(provider_dir / CUDA_COOP_TILE_INSTALLED_INCLUDE_RELATIVE);
      push_unique_existing_path(installed_headers, &out, &seen);
#endif
      if (out.empty())
      {
        append_cccl_include_paths(find_build_cccl_root(provider_dir), &out, &seen);
      }
    }
  }
  append_cuda_include_path(&out, &seen);

  fs::path selected;
  const bool selected_toolkit = configured_cuda_root(&selected).empty() && !selected.empty();
  // Explicit include path env vars are kept as an escape hatch.
  for (const char* env_key : {"CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS", "CUDA_COOP_TILE_INCLUDE_PATHS"})
  {
    const char* const env_val = std::getenv(env_key);
    if (!env_val || env_val[0] == '\0')
    {
      continue;
    }
    for (const std::string& part : split_encoded_path_list(env_val))
    {
      // Python serializes fallback CUDA headers along with the CCCL roots.
      // The selected toolkit already supplied its CUDA headers above.
      if (selected_toolkit
          && (is_file_path(fs::path(part) / "cuda_runtime.h") || is_file_path(fs::path(part) / "nvrtc.h")))
      {
        continue;
      }
      push_unique_existing_path(part, &out, &seen);
    }
  }

  return out;
}

void append_unique_existing_include_path(
  const std::string& path, std::vector<std::string>* out, std::unordered_set<std::string>* seen)
{
  push_unique_existing_path(path, out, seen);
}

bool include_paths_contain_file(const std::vector<std::string>& include_paths, const std::string& relative_path)
{
  for (const std::string& path : include_paths)
  {
    if (is_file_path(fs::path(path) / relative_path))
    {
      return true;
    }
  }
  return false;
}
} // namespace cuda::coop::tile::detail::provider
