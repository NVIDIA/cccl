#pragma once

#include <memory>
#include <string>
#include <vector>

#include <hostjit/compiler.hpp>
#include <hostjit/config.hpp>
#include <hostjit/loader.hpp>

namespace hostjit
{
class JITCompiler
{
public:
  // Create JIT compiler with default configuration (auto-detected)
  JITCompiler();

  // Create JIT compiler with custom configuration
  explicit JITCompiler(const CompilerConfig& config);

  ~JITCompiler();

  // Disable copy
  JITCompiler(const JITCompiler&)            = delete;
  JITCompiler& operator=(const JITCompiler&) = delete;

  // Compile CUDA source code to shared library and load it
  // Returns true on success, false on failure
  bool compile(const std::string& source_code);

  // Compile CUDA source code to a shared library WITHOUT loading it (no
  // dlopen/LoadLibrary). For the AoT compile/load split: the caller reads
  // the artifact bytes via getArtifactsPath() and persists them; loading
  // happens later, possibly in a different process, via loadFromBytes().
  // Returns true on success, false on failure.
  bool compileOnly(const std::string& source_code);

  // Load a previously-compiled shared library from an in-memory byte buffer
  // (as produced by compileOnly()+getArtifactsPath(), then persisted via
  // serialize()/deserialize()). Writes the bytes to a cache file and loads
  // that.
  //
  // On Linux, first resolves and dlopen(RTLD_GLOBAL)s this machine's own
  // libcudart.so.<major> (via findCudaRuntimeLibrary(config_)) *before*
  // loading the artifact. This is required for cross-machine portability:
  // the artifact's own baked RPATH points at the *build* machine's CUDA
  // Toolkit location, which may not exist here. Pre-resolving libcudart by
  // SONAME makes the artifact's dependency satisfied before any path
  // search (RPATH/RUNPATH/LD_LIBRARY_PATH/ld.so.cache) is ever consulted.
  // On Windows there is no equivalent baked-path problem (DLL dependencies
  // resolve by name via the standard search order), so no preload step is
  // needed there — the caller is responsible for ensuring the CUDA
  // Toolkit's bin directory (hosting cudart64_XX.dll) is on PATH.
  //
  // Returns true on success, false on failure (see getLastError()).
  bool loadFromBytes(const std::vector<char>& library_bytes);

  // Get function pointer by name
  // Returns nullptr if function not found
  template <typename FuncType>
  FuncType getFunction(const std::string& name)
  {
    if (!library_.isLoaded())
    {
      last_error_ = "No library loaded";
      return nullptr;
    }

    auto func = library_.getFunction<FuncType>(name);
    if (!func)
    {
      last_error_ = "Failed to find function '" + name + "': " + library_.getLastError();
    }
    return func;
  }

  // Get the last error message
  std::string getLastError() const
  {
    return last_error_;
  }

  // Get the configuration being used
  const CompilerConfig& getConfig() const
  {
    return config_;
  }

  // Check if a library is currently loaded
  bool isLoaded() const
  {
    return library_.isLoaded();
  }

  // Get the path to compiled artifacts (object file, shared library, etc.)
  // Only valid after successful compile() and if keep_artifacts is set
  std::string getArtifactsPath() const
  {
    return temp_dir_;
  }

  // Get the cubin extracted during compilation
  const std::vector<char>& getCubin() const
  {
    return cubin_;
  }

  // Unload the current library and clean up temporary files
  void cleanup();

private:
  std::string createTempDirectory();
  void removeTempDirectory();

  // Shared by compile() and compileOnly(): compile + link, no load.
  bool compileAndLink(const std::string& source_code, std::string* out_lib_path);

  CompilerConfig config_;
  CUDACompiler compiler_;
  DynamicLibrary library_;
  std::string temp_dir_;
  std::string last_error_;
  std::vector<char> cubin_;
};
} // namespace hostjit
