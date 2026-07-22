#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace hostjit
{
struct CompilerConfig
{
  std::string cuda_toolkit_path;
  std::string hostjit_include_path; // Path to hostjit include directory (for minimal CUDA runtime)
  std::string clang_headers_path; // Path to Clang's built-in CUDA headers (overrides CLANG_HEADERS_DIR)
  std::string cccl_include_path; // Path to CCCL headers (overrides CCCL_SOURCE_DIR); contains cub/, thrust/, cuda/
  std::vector<std::string> include_paths;
  std::vector<std::string> library_paths;
  std::vector<std::string> device_bitcode_files; // Raw LLVM bitcode (magic "BC") linked via LLVM's Linker
  std::vector<std::string> device_ltoir_files; // NVRTC LTOIR; linked at the nvJitLink stage with -lto
  std::unordered_map<std::string, std::string> macro_definitions; // key=macro name, value=macro value (empty for flag
                                                                  // macros)
  int sm_version         = 70;
  int optimization_level = 2;
  bool debug             = false;
  bool verbose           = false;
  bool trace_includes    = false; // Show all included headers during compilation (for debugging header search)
  bool keep_artifacts    = false; // Keep compiled artifacts for inspection (PTX, object files, etc.)
  std::string entry_point_name; // Name of the exported entry point function (used for post-link optimization)
  bool enable_pch = false; // Cache precompiled headers on disk to speed up repeated builds
};

// Auto-detect CUDA toolkit and create default configuration
CompilerConfig detectDefaultConfig();

// Validate that the configuration is usable
bool validateConfig(const CompilerConfig& config, std::string* error_message = nullptr);

// Locate the CUDA runtime shared library on this machine. Used both at link
// time (embedding it as an explicit linker input, since pip-installed CUDA
// Toolkits often lack an unversioned `libcudart.so` symlink that `-lcudart`
// needs) and at AoT load time (to pre-resolve the dependency by SONAME
// before dlopen'ing a persisted artifact whose baked RPATH may point at a
// different machine's CUDA Toolkit install. Returns an empty string if not
// found.
//
// Linux: returns a full path to a `libcudart.so*` file (found by scanning
// `config.library_paths`).
// Windows: returns a bare DLL filename (e.g. "cudart64_13.dll") — Windows
// resolves DLLs by name via the standard search order, not by baked path,
// so a full path isn't needed here.
std::string findCudaRuntimeLibrary(const CompilerConfig& config);
} // namespace hostjit
