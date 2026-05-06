#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>

namespace hostjit
{
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
