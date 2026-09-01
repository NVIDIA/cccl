#include <cstdlib>
#include <filesystem>
#include <string>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>

namespace hostjit
{
namespace
{
void append_system_include_path(std::vector<std::string>& args, const std::string& include_path)
{
  if (!include_path.empty())
  {
    args.push_back("--system-include-path=" + include_path);
  }
}

void append_macro_definition(
  std::vector<std::string>& args, const std::string& macro_name, const std::string& macro_value)
{
  if (macro_value.empty())
  {
    args.push_back("-D" + macro_name);
  }
  else
  {
    args.push_back("-D" + macro_name + "=" + macro_value);
  }
}

void append_cccl_include_paths(std::vector<std::string>& args, const CompilerConfig& config)
{
  if (!config.cccl_include_path.empty())
  {
    append_system_include_path(args, config.cccl_include_path);
    return;
  }

#ifdef CCCL_SOURCE_DIR
  append_system_include_path(args, std::string(CCCL_SOURCE_DIR) + "/libcudacxx/include");
  append_system_include_path(args, std::string(CCCL_SOURCE_DIR) + "/cub");
  append_system_include_path(args, std::string(CCCL_SOURCE_DIR) + "/thrust");
#endif
}

void append_cccl_macro_definitions(std::vector<std::string>& args)
{
  append_macro_definition(args, "CCCL_DISABLE_CTK_COMPATIBILITY_CHECK", "");
  append_macro_definition(args, "_CCCL_ENABLE_FREESTANDING", "1");
  append_macro_definition(args, "CCCL_DISABLE_NVTX", "1");
  append_macro_definition(args, "CCCL_DISABLE_EXCEPTIONS", "1");
}
} // namespace

void CompilerConfig::appendCommandLineArguments(std::vector<std::string>& args) const
{
  if (!cuda_toolkit_path.empty())
  {
    args.push_back("--cuda-path=" + cuda_toolkit_path);
  }
  if (!hostjit_include_path.empty())
  {
    args.push_back("--hostjit-include-path=" + hostjit_include_path);
  }
  if (!clang_headers_path.empty())
  {
    args.push_back("--clang-headers-path=" + clang_headers_path);
  }
  append_cccl_include_paths(args, *this);
  for (const auto& include_path : include_paths)
  {
    args.push_back("-I" + include_path);
  }
  for (const auto& library_path : library_paths)
  {
    args.push_back("-L" + library_path);
  }
  for (const auto& bitcode_file : device_bitcode_files)
  {
    args.push_back("--device-bitcode=" + bitcode_file);
  }
  for (const auto& ltoir_file : device_ltoir_files)
  {
    args.push_back("--device-ltoir=" + ltoir_file);
  }
  append_cccl_macro_definitions(args);
  for (const auto& [macro_name, macro_value] : macro_definitions)
  {
    append_macro_definition(args, macro_name, macro_value);
  }
  for (const auto& clang_arg : extra_clang_args)
  {
    args.push_back("-XClang");
    args.push_back(clang_arg);
  }
  if (!device_pch_path.empty())
  {
    args.push_back("--device-pch=" + device_pch_path);
  }
  if (!host_pch_path.empty())
  {
    args.push_back("--host-pch=" + host_pch_path);
  }
  args.push_back("--gpu-architecture=sm_" + std::to_string(sm_version));
  args.push_back("-O" + std::to_string(optimization_level));
  if (debug)
  {
    args.push_back("--debug");
  }
  if (verbose)
  {
    args.push_back("--verbose");
  }
  if (trace_includes)
  {
    args.push_back("--trace-includes");
  }
  if (keep_artifacts)
  {
    args.push_back("--keep-artifacts");
  }
  if (!entry_point_name.empty())
  {
    args.push_back("--entry-point=" + entry_point_name);
  }
}

CompilerConfig detectDefaultConfig()
{
  CompilerConfig config;

  // Detect CUDA toolkit path
  if (const char* env = std::getenv("CUDA_PATH"))
  {
    config.cuda_toolkit_path = env;
  }
  else if (const char* env = std::getenv("CUDA_HOME"))
  {
    config.cuda_toolkit_path = env;
  }
#ifdef CUDA_TOOLKIT_PATH
  else
  {
    config.cuda_toolkit_path = CUDA_TOOLKIT_PATH;
  }
#endif

  // Set up library paths if CUDA toolkit was found
  if (!config.cuda_toolkit_path.empty())
  {
    std::filesystem::path lib64_path = std::filesystem::path(config.cuda_toolkit_path) / "lib64";
    std::filesystem::path lib_path   = std::filesystem::path(config.cuda_toolkit_path) / "lib";

    if (std::filesystem::exists(lib64_path))
    {
      config.library_paths.push_back(lib64_path.string());
    }
    else if (std::filesystem::exists(lib_path))
    {
      config.library_paths.push_back(lib_path.string());
    }
  }

  // Auto-detect GPU compute capability using CUDA runtime
  int device = 0;
  if (cudaGetDevice(&device) == cudaSuccess)
  {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
    {
      int detected_sm = prop.major * 10 + prop.minor;
      if (detected_sm >= 75)
      {
        config.sm_version = detected_sm;
      }
    }
  }

  if (config.sm_version == 0)
  {
    config.sm_version = 75;
  }

  config.optimization_level = 2;
  config.debug              = false;
  config.verbose            = false;

  // Detect hostjit include path
  if (const char* env = std::getenv("HOSTJIT_INCLUDE_PATH"))
  {
    config.hostjit_include_path = env;
  }
#ifdef HOSTJIT_INCLUDE_DIR
  else
  {
    config.hostjit_include_path = HOSTJIT_INCLUDE_DIR;
  }
#endif

  // Detect clang headers path. Build-time CLANG_HEADERS_DIR is the default;
  // HOSTJIT_CLANG_PATH overrides it (e.g. for pip-installed wheels with a
  // packaged copy of clang's CUDA headers).
  if (const char* env = std::getenv("HOSTJIT_CLANG_PATH"))
  {
    config.clang_headers_path = env;
  }
#ifdef CLANG_HEADERS_DIR
  else
  {
    config.clang_headers_path = CLANG_HEADERS_DIR;
  }
#endif

  return config;
}

bool validateConfig(const CompilerConfig& config, std::string* error_message)
{
  if (config.cuda_toolkit_path.empty())
  {
    if (error_message)
    {
      *error_message = "CUDA toolkit path not found. Please set CUDA_PATH or CUDA_HOME environment variable.";
    }
    return false;
  }

  if (!std::filesystem::exists(config.cuda_toolkit_path))
  {
    if (error_message)
    {
      *error_message = "CUDA toolkit path does not exist: " + config.cuda_toolkit_path;
    }
    return false;
  }

  std::filesystem::path cuda_h = std::filesystem::path(config.cuda_toolkit_path) / "include" / "cuda.h";
  if (!std::filesystem::exists(cuda_h))
  {
    if (error_message)
    {
      *error_message = "CUDA headers not found at: " + cuda_h.string();
    }
    return false;
  }

  for (const auto& include_path : config.include_paths)
  {
    if (!std::filesystem::exists(include_path))
    {
      if (error_message)
      {
        *error_message = "Include path does not exist: " + include_path;
      }
      return false;
    }
  }

  for (const auto& library_path : config.library_paths)
  {
    if (!std::filesystem::exists(library_path))
    {
      if (error_message)
      {
        *error_message = "Library path does not exist: " + library_path;
      }
      return false;
    }
  }

  if (config.sm_version < 30 || config.sm_version > 150)
  {
    if (error_message)
    {
      *error_message = "Invalid SM version: " + std::to_string(config.sm_version) + " (must be between 30 and 150)";
    }
    return false;
  }

  if (config.optimization_level < 0 || config.optimization_level > 3)
  {
    if (error_message)
    {
      *error_message =
        "Invalid optimization level: " + std::to_string(config.optimization_level) + " (must be between 0 and 3)";
    }
    return false;
  }

  return true;
}
} // namespace hostjit
