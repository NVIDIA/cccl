#include "kernel.cuh"

void b()
{
  int* d_out{};
  cudaMalloc(&d_out, sizeof(int));
  cudaMemset(d_out, 0, sizeof(int));

  void* ptr = reinterpret_cast<void*>(kernel<int>);

  printf("b: kernel stub address: %p\n", ptr);

  cudaFunction_t func{};
  if (cudaError_t error = cudaGetFuncBySymbol(&func, ptr))
  {
    printf("b: kernel NOT found in mapping: %s\n", cudaGetErrorString(error));
  }
  else
  {
    printf("b: kernel is in mapping: %s\n", cudaGetErrorString(error));
  }

  kernel<<<1, 1>>>('b', d_out);

  if (cudaPeekAtLastError() != cudaSuccess)
  {
    printf("b: FAILED to launch kernel\n");
  }
  else
  {
    printf("b: launched kernel\n");
  }

  if (cudaStreamSynchronize(0) != cudaSuccess)
  {
    printf("b: FAILED to synchronize stream\n");
  }
  else
  {
    printf("b: synchronized stream\n");
  }

  int h_out{};
  if (cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    printf("b: FAILED to copy from device to host\n");
  }
  else
  {
    printf("b: copied from device to host\n");
  }

  printf("b: out: %d\n", h_out);

  if (h_out != 42)
  {
    printf("b: kernel was NOT actually launched: out != 42\n");
  }
  else
  {
    printf("b: kernel was launched: out == 42\n");
  }

  printf("\n");
}
