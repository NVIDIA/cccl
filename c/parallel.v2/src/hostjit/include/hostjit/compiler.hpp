#pragma once

#include <string>
#include <vector>

namespace hostjit
{
struct CompilationResult
{
  bool success;
  std::string object_file_path; // Path to generated .o file
  std::string diagnostics; // Compiler messages
  std::vector<char> cubin; // Device cubin extracted during compilation
};

struct BitcodeResult
{
  bool success;
  std::string bitcode; // LLVM bitcode bytes
  std::string diagnostics;
};

struct LinkResult
{
  bool success;
  std::string library_path; // Path to .so file
  std::string diagnostics;
};

// Forward declaration to avoid including heavy Clang headers
struct CompilerConfig;

class CUDACompiler
{
public:
  CUDACompiler();
  ~CUDACompiler();

  // Compile CUDA device source to LLVM bitcode
  BitcodeResult compileToDeviceBitcode(const std::string& source_code, const CompilerConfig& config);

  // Compile CUDA source code to object file
  CompilationResult
  compileToObject(const std::string& source_code, const std::string& output_path, const CompilerConfig& config);

  // Link object files to shared library
  LinkResult linkToSharedLibrary(
    const std::vector<std::string>& object_files, const std::string& output_path, const CompilerConfig& config);

private:
  class Impl;
  Impl* impl_;
};
} // namespace hostjit
