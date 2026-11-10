// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cctype>
#include <climits>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nvrtc.h>

#include "provider/runtime/runtime.h"

namespace cuda::coop::tile::detail::provider
{
namespace
{
// Own a single NVRTC program through every early-return path.
// Diagnostics are copied while the handle is still valid. Output bytes are
// copied into caller-owned storage before this guard destroys the program.
struct nvrtc_program
{
  nvrtcProgram handle = nullptr;

  nvrtc_program()                                = default;
  nvrtc_program(const nvrtc_program&)            = delete;
  nvrtc_program& operator=(const nvrtc_program&) = delete;

  ~nvrtc_program() noexcept
  {
    if (handle)
    {
      nvrtcDestroyProgram(&handle);
    }
  }
};

#define CUDA_COOP_TILE_NVRTC_CHECK(call, prog, out_error, on_error) \
  do                                                                \
  {                                                                 \
    const nvrtcResult result = (call);                              \
    if (result != NVRTC_SUCCESS)                                    \
    {                                                               \
      record_nvrtc_failure(#call, result, prog, out_error);         \
      on_error;                                                     \
    }                                                               \
  } while (false)

void record_nvrtc_failure(const char* expression, nvrtcResult result, nvrtcProgram prog, std::string* out_error)
{
  std::ostringstream detail;
  detail << "[LP-PROVIDER-INTERNAL-001] NVRTC error: " << expression << " rc=" << static_cast<int>(result);
  if (prog)
  {
    size_t log_size = 0;
    if (nvrtcGetProgramLogSize(prog, &log_size) == NVRTC_SUCCESS && log_size > 1)
    {
      std::vector<char> log(log_size);
      if (nvrtcGetProgramLog(prog, log.data()) == NVRTC_SUCCESS)
      {
        detail << '\n' << log.data();
      }
    }
  }
  if (out_error)
  {
    *out_error = detail.str();
  }
  emit_trace_line("[coop_linkage_provider][trace] " + detail.str());
}

void trace_nvrtc_target_option(const char* operation, const std::string& unit_name, TargetArchitecture target_arch)
{
  std::ostringstream oss;
  oss << "[coop_linkage_provider][trace] " << operation << " unit=" << unit_name
      << " option=--gpu-architecture=" << target_arch.compute_name();
  emit_trace_line(oss.str());
}

// Read the compile-time resource table from NVRTC's PTX output.
// Decode byte, 32-bit, or 64-bit initializer storage, including omitted trailing
// zero components. The sentinel and version distinguish this table from an
// unrelated constant before its size/alignment words are returned. No GPU
// kernel is launched to obtain these values.
[[nodiscard]] bool
parse_ptx_resource_sizes(const std::string& ptx, size_t expected_count, std::vector<uint64_t>* out_sizes)
{
  if (!out_sizes || expected_count == 0)
  {
    return false;
  }
  out_sizes->clear();

  constexpr char symbol[] = "cuda_coop_tile_resource_sizes";
  const size_t symbol_pos = ptx.find(symbol);
  if (symbol_pos == std::string::npos)
  {
    emit_trace_line("[coop_linkage_provider][trace] resource query PTX did not contain symbol " + std::string(symbol));
    return false;
  }

  size_t stmt_start     = ptx.rfind('\n', symbol_pos);
  stmt_start            = (stmt_start == std::string::npos) ? 0 : stmt_start + 1;
  const size_t stmt_end = ptx.find(';', symbol_pos);
  if (stmt_end == std::string::npos)
  {
    emit_trace_line("[coop_linkage_provider][trace] resource query PTX did not contain a complete initializer for "
                    + std::string(symbol));
    return false;
  }
  const std::string stmt = ptx.substr(stmt_start, stmt_end - stmt_start);
  const size_t eq        = stmt.find('=');
  if (eq == std::string::npos)
  {
    emit_trace_line(
      "[coop_linkage_provider][trace] resource query PTX initializer has no '=' for " + std::string(symbol));
    return false;
  }

  const std::string init = stmt.substr(eq + 1);
  std::vector<uint64_t> values;
  const char* cursor = init.c_str();
  while (*cursor != '\0')
  {
    while (*cursor != '\0' && !std::isdigit(static_cast<unsigned char>(*cursor)))
    {
      ++cursor;
    }
    if (*cursor == '\0')
    {
      break;
    }
    char* end                       = nullptr;
    const unsigned long long parsed = std::strtoull(cursor, &end, 0);
    if (end == cursor)
    {
      ++cursor;
      continue;
    }
    values.push_back(static_cast<uint64_t>(parsed));
    cursor = end;
  }

  const size_t expected_words = expected_count + resource_query_header_words;
  const bool stores_bytes     = stmt.find(".b8") != std::string::npos || stmt.find(".u8") != std::string::npos;
  const bool stores_word32    = stmt.find(".b32") != std::string::npos || stmt.find(".u32") != std::string::npos;
  const bool stores_word64    = stmt.find(".b64") != std::string::npos || stmt.find(".u64") != std::string::npos;
  const int storage_modes =
    static_cast<int>(stores_bytes) + static_cast<int>(stores_word32) + static_cast<int>(stores_word64);
  if (storage_modes != 1)
  {
    emit_trace_line("[coop_linkage_provider][trace] resource query PTX initializer uses unsupported storage for "
                    + std::string(symbol));
    return false;
  }

  if (stores_bytes || stores_word32)
  {
    const size_t component_bits      = stores_bytes ? 8 : 32;
    const size_t components_per_word = sizeof(uint64_t) / (component_bits / CHAR_BIT);
    const size_t required_components = expected_words * components_per_word;
    if (values.size() < required_components)
    {
      // PTX permits braced initializers to omit trailing zero components.
      values.resize(required_components, 0);
    }
    const uint64_t max_component = stores_bytes ? 0xffULL : 0xffffffffULL;
    std::vector<uint64_t> decoded_words;
    decoded_words.reserve(expected_words);
    for (size_t i = 0; i < expected_words; ++i)
    {
      uint64_t value = 0;
      for (size_t component_index = 0; component_index < components_per_word; ++component_index)
      {
        const uint64_t component = values[i * components_per_word + component_index];
        if (component > max_component)
        {
          std::ostringstream detail;
          detail << "[coop_linkage_provider][trace] resource query PTX " << component_bits
                 << "-bit initializer contains out-of-range value=" << component;
          emit_trace_line(detail.str());
          return false;
        }
        value |= component << (component_bits * component_index);
      }
      decoded_words.push_back(value);
    }
    values = std::move(decoded_words);
  }

  if (values.size() < expected_words)
  {
    std::ostringstream detail;
    detail << "[coop_linkage_provider][trace] resource query PTX initializer too small: got " << values.size()
           << " values expected " << expected_words;
    emit_trace_line(detail.str());
    return false;
  }

  if (values[0] != resource_query_sentinel || values[1] != resource_query_version)
  {
    std::ostringstream detail;
    detail << "[coop_linkage_provider][trace] resource query PTX header mismatch: sentinel=0x" << std::hex << values[0]
           << std::dec << " version=" << values[1];
    emit_trace_line(detail.str());
    return false;
  }

  out_sizes->assign(values.begin() + resource_query_header_words,
                    values.begin() + resource_query_header_words + expected_count);
  return true;
}
} // namespace

// Compile generated device source and copy its LTO-IR into out_ltoir.
// Option strings remain alive while NVRTC borrows their c_str pointers. The
// output buffer is cleared on entry, but callers must use it only on success:
// a failure while copying the result can leave a resized, incomplete vector.
// The program guard cleans up after both compiler and extraction failures.
bool compile_source_to_ltoir(
  const std::string& source,
  const std::string& entry_name,
  TargetArchitecture target_arch,
  int resolved_block_threads,
  int resolved_items_per_thread,
  const std::vector<std::string>& include_paths,
  std::vector<char>* out_ltoir,
  std::string* out_error)
{
  if (out_error)
  {
    out_error->clear();
  }
  if (out_ltoir)
  {
    out_ltoir->clear();
  }
  if (!out_ltoir)
  {
    if (out_error)
    {
      *out_error = "[LP-PROVIDER-INTERNAL-001] provider compilation requires an output LTO-IR buffer";
    }
    return false;
  }

  if (!target_arch.valid())
  {
    if (out_error)
    {
      *out_error = "[LP-PROVIDER-INTERNAL-001] provider compilation requires resolved target architecture metadata";
    }
    return false;
  }

  std::vector<std::string> opt_storage;
  opt_storage.emplace_back("--std=c++17");
  opt_storage.emplace_back("--gpu-architecture=" + target_arch.compute_name());
  opt_storage.emplace_back("--relocatable-device-code=true");
  opt_storage.emplace_back("-dlto");
  if (resolved_block_threads > 0)
  {
    opt_storage.emplace_back("-DCUDA_COOP_TILE_BLOCK_THREADS=" + std::to_string(resolved_block_threads));
  }
  if (resolved_items_per_thread > 0)
  {
    opt_storage.emplace_back("-DCUDA_COOP_TILE_ITEMS_PER_THREAD=" + std::to_string(resolved_items_per_thread));
  }
  for (const auto& path : include_paths)
  {
    opt_storage.emplace_back("--include-path=" + path);
  }

  std::vector<const char*> opts;
  opts.reserve(opt_storage.size());
  for (const auto& opt : opt_storage)
  {
    opts.push_back(opt.c_str());
  }

  const std::string unit_name = entry_name.empty() ? "coop_provider_generated.cu" : (entry_name + ".cu");
  trace_nvrtc_target_option("nvrtc_lto_compile", unit_name, target_arch);

  nvrtc_program program;
  auto& prog = program.handle;
  CUDA_COOP_TILE_NVRTC_CHECK(
    nvrtcCreateProgram(&prog, source.c_str(), unit_name.c_str(), 0, nullptr, nullptr), prog, out_error, return false);

  CUDA_COOP_TILE_NVRTC_CHECK(
    nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data()), prog, out_error, return false);

  size_t lto_size = 0;
  CUDA_COOP_TILE_NVRTC_CHECK(nvrtcGetLTOIRSize(prog, &lto_size), prog, out_error, return false);

  out_ltoir->resize(lto_size);
  CUDA_COOP_TILE_NVRTC_CHECK(nvrtcGetLTOIR(prog, out_ltoir->data()), prog, out_error, return false);

  return true;
}

// Compile a constant size table to PTX and decode it without device execution.
// Use the requested target and include order so sizeof/alignof match the
// wrapper compilation. Only a complete validated table is a successful result;
// NVRTC logs or table-decoding errors are returned through out_error.
bool compile_resource_query_to_sizes(
  const std::string& source,
  TargetArchitecture target_arch,
  const std::vector<std::string>& include_paths,
  size_t expected_count,
  std::vector<uint64_t>* out_sizes,
  std::string* out_error)
{
  if (out_error)
  {
    out_error->clear();
  }
  if (out_sizes)
  {
    out_sizes->clear();
  }
  if (!out_sizes || expected_count == 0)
  {
    if (out_error)
    {
      *out_error = "[LP-PROVIDER-INTERNAL-001] invalid resource-query output contract";
    }
    return false;
  }

  if (!target_arch.valid())
  {
    if (out_error)
    {
      *out_error = "[LP-PROVIDER-INTERNAL-001] resource query has no resolved target architecture";
    }
    emit_trace_line("[coop_linkage_provider][trace] resource query has no resolved target architecture");
    return false;
  }

  std::vector<std::string> opt_storage;
  opt_storage.emplace_back("--std=c++17");
  opt_storage.emplace_back("--gpu-architecture=" + target_arch.compute_name());
  for (const auto& path : include_paths)
  {
    opt_storage.emplace_back("--include-path=" + path);
  }

  std::vector<const char*> opts;
  opts.reserve(opt_storage.size());
  for (const auto& opt : opt_storage)
  {
    opts.push_back(opt.c_str());
  }

  trace_nvrtc_target_option("nvrtc_resource_compile", "coop_resource_query.cu", target_arch);
  nvrtc_program program;
  auto& prog = program.handle;
  CUDA_COOP_TILE_NVRTC_CHECK(
    nvrtcCreateProgram(&prog, source.c_str(), "coop_resource_query.cu", 0, nullptr, nullptr),
    prog,
    out_error,
    return false);

  CUDA_COOP_TILE_NVRTC_CHECK(
    nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data()), prog, out_error, return false);

  size_t ptx_size = 0;
  CUDA_COOP_TILE_NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size), prog, out_error, return false);
  std::string ptx(ptx_size, '\0');
  CUDA_COOP_TILE_NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()), prog, out_error, return false);
  if (parse_ptx_resource_sizes(ptx, expected_count, out_sizes))
  {
    return true;
  }
  if (out_error)
  {
    *out_error = "[LP-PROVIDER-INTERNAL-001] NVRTC resource-query PTX did not contain the expected size data";
  }
  return false;
}
} // namespace cuda::coop::tile::detail::provider
