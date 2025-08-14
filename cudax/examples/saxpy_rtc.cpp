#include <cuda/experimental/compiler.cuh>

#include <cstdlib>
#include <iostream>

#include <cuda.h>

#define CUDA_SAFE_CALL(x)                                               \
  do                                                                    \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS)                                         \
    {                                                                   \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      std::cerr << "\nerror: " #x " failed with error " << msg << '\n'; \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

namespace cudax = cuda::experimental;

constexpr int num_threads = 128;
constexpr int num_blocks  = 128;

constexpr auto saxpy = "                                        \n\
extern \"C\" __global__                                         \n\
void saxpy(float a, float *x, float *y, float *out, size_t n)   \n\
{                                                               \n\
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;           \n\
  if (tid < n) {                                                \n\
    out[tid] = a * x[tid] + y[tid];                             \n\
  }                                                             \n\
}                                                               \n";

int main()
{
  cudax::cuda_compile_source src{"saxpy.cu", saxpy};
  cudax::cuda_compile_options cuda_opts{};
  cuda_opts.enable_fmad(false); // Compile the program with fmad disabled.

  cudax::cuda_compiler compiler{};
  const auto compile_result = compiler.compile_to_ptx(src, cuda_opts);
  const auto log            = compile_result.log();
  std::cout << log << '\n';

  if (!compile_result)
  {
    std::exit(1);
  }

  const auto ptx = compile_result.ptx();

  // Load the generated PTX and get a handle to the SAXPY kernel.
  CUdevice device;
  CUcontext context;
  CUlibrary library;
  CUkernel kernel;

  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&device, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, nullptr, 0, device));
  CUDA_SAFE_CALL(cuLibraryLoadData(&library, ptx.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  CUDA_SAFE_CALL(cuLibraryGetKernel(&kernel, library, "saxpy"));

  // Generate input for execution, and create output buffers.
  size_t n          = num_threads * num_blocks;
  size_t bufferSize = n * sizeof(float);
  float a           = 5.1f;

  float* hX   = new float[n];
  float* hY   = new float[n];
  float* hOut = new float[n];
  for (size_t i = 0; i < n; ++i)
  {
    hX[i] = static_cast<float>(i);
    hY[i] = static_cast<float>(i * 2);
  }

  CUdeviceptr dX, dY, dOut;
  CUDA_SAFE_CALL(cuMemAlloc(&dX, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&dY, bufferSize));
  CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dY, hY, bufferSize));

  // Execute SAXPY.
  void* args[] = {&a, &dX, &dY, &dOut, &n};
  CUDA_SAFE_CALL(cuLaunchKernel(
    (CUfunction) kernel,
    num_blocks,
    1,
    1, // grid dim
    num_threads,
    1,
    1, // block dim
    0,
    NULL, // shared mem and stream
    args,
    0)); // arguments
  CUDA_SAFE_CALL(cuCtxSynchronize());
  // Retrieve and print output.
  CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));
  for (size_t i = 0; i < n; ++i)
  {
    std::cout << a << " * " << hX[i] << " + " << hY[i] << " = " << hOut[i] << '\n';
  }

  // Release resources.
  CUDA_SAFE_CALL(cuMemFree(dX));
  CUDA_SAFE_CALL(cuMemFree(dY));
  CUDA_SAFE_CALL(cuMemFree(dOut));
  CUDA_SAFE_CALL(cuLibraryUnload(library));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
  delete[] hX;
  delete[] hY;
  delete[] hOut;
}
