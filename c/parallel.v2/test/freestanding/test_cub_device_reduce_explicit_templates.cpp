// Repro harness: feeds a CUDA source string to v1's hostjit and reports
// whether compilation succeeds. If a path is passed via argv[1] or
// $REPRO_SOURCE_FILE, that file's contents are compiled instead of the
// built-in minimal source. Used to test whether v2's actual CubCall-generated
// host_input.cu compiles under v1's hostjit infrastructure.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

#include <hostjit/config.hpp>
#include <hostjit/jit_compiler.hpp>

static const char* default_source = R"(
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

using in_0_it_t  = int*;
using out_0_it_t = unsigned long long*;

struct Op_0 {
    __device__ __forceinline__
    unsigned long long operator()(unsigned long long a, unsigned long long b) const {
        return a + b;
    }
};

extern "C" __attribute__((visibility("default"))) int cccl_jit_reduce(
    void* d_temp_storage,
    size_t* temp_storage_bytes,
    void* d_in_state,
    void* d_out_state,
    unsigned long long num_items,
    void* /*op_state*/,
    void* init_state)
{
    in_0_it_t  d_in     = static_cast<in_0_it_t>(d_in_state);
    out_0_it_t d_out    = static_cast<out_0_it_t>(d_out_state);
    unsigned long long init = *static_cast<unsigned long long*>(init_state);
    Op_0 op;
    cudaError_t err = cub::DeviceReduce::Reduce<in_0_it_t, out_0_it_t, Op_0, int, unsigned long long>(
        d_temp_storage, *temp_storage_bytes, d_in, d_out,
        static_cast<int>(num_items), op, init);
    return err == cudaSuccess ? 0 : -1;
}
)";

int main(int argc, char** argv)
{
  std::string source_str;
  std::string source_path;

  if (argc > 1)
  {
    source_path = argv[1];
  }
  else if (const char* env = std::getenv("REPRO_SOURCE_FILE"))
  {
    source_path = env;
  }

  if (!source_path.empty())
  {
    std::ifstream f(source_path);
    if (!f)
    {
      std::cerr << "Failed to open: " << source_path << std::endl;
      return 2;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    source_str = ss.str();
    std::cerr << "Loaded " << source_str.size() << " bytes from " << source_path << std::endl;
  }
  else
  {
    source_str = default_source;
    std::cerr << "Using built-in default source." << std::endl;
  }

  hostjit::CompilerConfig config = hostjit::detectDefaultConfig();
  config.sm_version              = 80;
  config.verbose                 = false;

  hostjit::JITCompiler compiler(config);
  if (!compiler.compile(source_str))
  {
    std::cerr << "JIT compilation FAILED:\n" << compiler.getLastError() << std::endl;
    return 1;
  }

  std::cout << "JIT compilation succeeded." << std::endl;
  return 0;
}
