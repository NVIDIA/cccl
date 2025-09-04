#include "kernel.cuh"

void a()
{
  int* d_out{};
  cudaMalloc(&d_out, sizeof(int));
  cudaMemset(d_out, 0, sizeof(int));

  void* ptr = reinterpret_cast<void*>(kernel<int>);

  std::printf("a: kernel stub address: %p\n", ptr);

  cudaFunction_t func{};
  if (cudaError_t error = cudaGetFuncBySymbol(&func, ptr))
  {
    std::printf("a: kernel NOT found in mapping: %s\n", cudaGetErrorString(error));
  }
  else
  {
    std::printf("a: kernel is in mapping: %s\n", cudaGetErrorString(error));
  }

  kernel<<<1, 1>>>('a', d_out);

  if (cudaPeekAtLastError() != cudaSuccess)
  {
    std::printf("b: FAILED to launch kernel\n");
  }
  else
  {
    std::printf("b: launched kernel\n");
  }

  if (cudaStreamSynchronize(0) != cudaSuccess)
  {
    std::printf("a: FAILED to synchronize stream\n");
  }
  else
  {
    std::printf("a: synchronized stream\n");
  }

  int h_out{};
  if (cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    std::printf("a: FAILED to copy from device to host\n");
  }
  else
  {
    std::printf("a: copied from device to host\n");
  }

  std::printf("a: out: %d\n", h_out);
  if (h_out != 42)
  {
    std::printf("a: kernel was NOT actually launched: out != 42\n");
  }
  else
  {
    std::printf("a: kernel was launched: out == 42\n");
  }

  std::printf("\n");
}
