#pragma once

#include <memory>
#include <string>

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

  CompilerConfig config_;
  CUDACompiler compiler_;
  DynamicLibrary library_;
  std::string temp_dir_;
  std::string last_error_;
  std::vector<char> cubin_;
};
} // namespace hostjit
