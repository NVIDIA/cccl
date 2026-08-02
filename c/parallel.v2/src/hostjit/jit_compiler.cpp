#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include <hostjit/jit_compiler.hpp>

#ifdef _WIN32
#  include <process.h>
#else
#  include <dlfcn.h>
#  include <unistd.h>
#endif

namespace hostjit
{
JITCompiler::JITCompiler()
    : config_(detectDefaultConfig())
{}

JITCompiler::JITCompiler(const CompilerConfig& config)
    : config_(config)
{}

JITCompiler::~JITCompiler()
{
  cleanup();
}

// Shared by compile() and compileOnly(): validate config, compile to an
// object file, link to a shared library in a fresh temp dir. Does NOT load
// it. On success, *out_lib_path is the path to the linked artifact (still
// on disk in temp_dir_, which the caller may read before it's torn down by
// a later cleanup()).
bool JITCompiler::compileAndLink(const std::string& source_code, std::string* out_lib_path)
{
  std::string config_error;
  if (!validateConfig(config_, &config_error))
  {
    last_error_ = "Configuration error: " + config_error;
    return false;
  }

  cleanup();

  temp_dir_ = createTempDirectory();
  if (temp_dir_.empty())
  {
    last_error_ = "Failed to create temporary directory";
    return false;
  }

  std::string obj_path = temp_dir_ + "/cuda_code.o";
  auto compile_result  = compiler_.compileToObject(source_code, obj_path, config_);

  if (!compile_result.success)
  {
    last_error_ = "Compilation failed:\n" + compile_result.diagnostics;
    removeTempDirectory();
    return false;
  }

  // Store the cubin for later inspection
  cubin_ = std::move(compile_result.cubin);

  if (config_.verbose)
  {
    std::cout << "Compilation diagnostics:\n" << compile_result.diagnostics << "\n";
  }

#ifdef _WIN32
  std::string lib_path = temp_dir_ + "/cuda_code.dll";
#else
  std::string lib_path = temp_dir_ + "/libcuda_code.so";
#endif
  auto link_result = compiler_.linkToSharedLibrary({obj_path}, lib_path, config_);

  if (!link_result.success)
  {
    last_error_ = "Linking failed:\n" + link_result.diagnostics;
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Linking diagnostics:\n" << link_result.diagnostics << "\n";
  }

  *out_lib_path = lib_path;
  return true;
}

bool JITCompiler::compile(const std::string& source_code)
{
  std::string lib_path;
  if (!compileAndLink(source_code, &lib_path))
  {
    return false;
  }

  if (!library_.load(lib_path))
  {
    last_error_ = "Failed to load library: " + library_.getLastError();
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Successfully loaded library: " << lib_path << "\n";
  }

  last_error_.clear();
  return true;
}

bool JITCompiler::compileOnly(const std::string& source_code)
{
  std::string lib_path;
  if (!compileAndLink(source_code, &lib_path))
  {
    return false;
  }
  // Deliberately do NOT dlopen/LoadLibrary — the caller only wants the
  // artifact bytes (via getArtifactsPath()), e.g. to persist them for AoT.
  // temp_dir_ (and the artifact within it) stays on disk until the next
  // compile()/compileOnly()/loadFromBytes() call or destruction — the
  // caller must read the bytes before then.
  last_error_.clear();
  return true;
}

bool JITCompiler::loadFromBytes(const std::vector<char>& library_bytes)
{
  std::string config_error;
  if (!validateConfig(config_, &config_error))
  {
    last_error_ = "Configuration error: " + config_error;
    return false;
  }

  cleanup();

  temp_dir_ = createTempDirectory();
  if (temp_dir_.empty())
  {
    last_error_ = "Failed to create temporary directory";
    return false;
  }

#ifdef _WIN32
  std::string lib_path = temp_dir_ + "/cuda_code.dll";
#else
  std::string lib_path = temp_dir_ + "/libcuda_code.so";
#endif

  {
    std::ofstream out(lib_path, std::ios::binary);
    if (!out)
    {
      last_error_ = "Failed to open " + lib_path + " for writing";
      removeTempDirectory();
      return false;
    }
    out.write(library_bytes.data(), static_cast<std::streamsize>(library_bytes.size()));
    if (!out)
    {
      last_error_ = "Failed to write artifact bytes to " + lib_path;
      removeTempDirectory();
      return false;
    }
  }

#ifndef _WIN32
  // Cross-machine portability: pre-resolve and dlopen(RTLD_GLOBAL) this
  // machine's own libcudart.so.<major> *before* loading the artifact. The
  // artifact's own baked RPATH points at the machine it was BUILT on,
  // which may not have CUDA at that same path (e.g. a different pip
  // install layout). Because glibc's dynamic loader checks already-mapped
  // objects by SONAME before doing any path search (RPATH/RUNPATH/
  // LD_LIBRARY_PATH/ld.so.cache/default paths), pre-loading the correct
  // libcudart here satisfies the artifact's dependency regardless of what
  // its baked RPATH says.
  std::string cudart_path = findCudaRuntimeLibrary(config_);
  if (!cudart_path.empty())
  {
    dlerror();
    void* cudart_handle = dlopen(cudart_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!cudart_handle)
    {
      const char* err = dlerror();
      last_error_     = "Failed to preload CUDA runtime (" + cudart_path + "): " + (err ? err : "unknown dlopen error");
      removeTempDirectory();
      return false;
    }
    // Intentionally never dlclose'd (RTLD_GLOBAL; matches the "never
    // dlclose a JIT module" policy already in loader.cpp, see #9367 —
    // dlclose'ing here would risk unmapping libcudart while other loaded
    // JIT modules in this process still reference it).
  }
  // If cudart_path is empty (couldn't find any CUDA Toolkit on this
  // machine at all), fall through and let the subsequent load() fail
  // naturally with a clear "cannot open shared object" error.
#endif

  if (!library_.load(lib_path))
  {
    last_error_ = "Failed to load library: " + library_.getLastError();
    removeTempDirectory();
    return false;
  }

  last_error_.clear();
  return true;
}

void JITCompiler::cleanup()
{
  library_.unload();

  if (!config_.keep_artifacts)
  {
    removeTempDirectory();
  }

  last_error_.clear();
}

std::string JITCompiler::createTempDirectory()
{
  std::filesystem::path base_tmp_dir;

#ifdef _WIN32
  const char* tmp_dir = std::getenv("TEMP");
  if (!tmp_dir)
  {
    tmp_dir = std::getenv("TMP");
  }
  if (tmp_dir)
  {
    base_tmp_dir = tmp_dir;
  }
  else
  {
    base_tmp_dir = std::filesystem::temp_directory_path();
  }
#else
  const char* tmp_dir = std::getenv("TMPDIR");
  if (tmp_dir)
  {
    base_tmp_dir = tmp_dir;
  }
  else
  {
    base_tmp_dir = "/tmp";
  }
#endif

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 999999);

#ifdef _WIN32
  int pid = _getpid();
#else
  int pid = getpid();
#endif

  for (int attempt = 0; attempt < 10; ++attempt)
  {
    std::string dir_name            = "hostjit_" + std::to_string(pid) + "_" + std::to_string(dis(gen));
    std::filesystem::path full_path = base_tmp_dir / dir_name;

    std::error_code ec;
    if (std::filesystem::create_directories(full_path, ec) && !ec)
    {
      return full_path.string();
    }
  }

  return "";
}

void JITCompiler::removeTempDirectory()
{
  if (temp_dir_.empty())
  {
    return;
  }

  try
  {
    if (std::filesystem::exists(temp_dir_))
    {
      std::filesystem::remove_all(temp_dir_);
    }
  }
  catch (const std::filesystem::filesystem_error& e)
  {
    if (config_.verbose)
    {
      std::cerr << "Warning: Failed to remove temporary directory: " << e.what() << "\n";
    }
  }

  temp_dir_.clear();
}
} // namespace hostjit
