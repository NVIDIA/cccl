#include "kernel.cuh"

void b_launch(void (*k)(char, size_t*), char c, size_t* d_out);

void a_launch(void (*k)(char, int*), char c, int* d_out)
{
  void* ptr = reinterpret_cast<void*>(k);

  printf("a: kernel stub address: %p\n", ptr);

  cudaFunction_t func{};
  if (cudaError_t error = cudaGetFuncBySymbol(&func, ptr))
  {
    printf("a: kernel NOT found in mapping: %s\n", cudaGetErrorString(error));
  }
  else
  {
    printf("a: kernel is in mapping: %s\n", cudaGetErrorString(error));
  }

  cudaMemset(d_out, 0, sizeof(int));
  k<<<1, 1>>>(c, d_out);

  if (cudaPeekAtLastError() != cudaSuccess)
  {
    printf("a: FAILED to launch kernel\n");
  }
  else
  {
    printf("a: launched kernel\n");
  }

  if (cudaStreamSynchronize(0) != cudaSuccess)
  {
    printf("a: FAILED to synchronize stream\n");
  }
  else
  {
    printf("a: synchronized stream\n");
  }

  int h_out{};
  if (cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    printf("a: FAILED to copy from device to host\n");
  }
  else
  {
    printf("a: copied from device to host\n");
  }

  printf("a: out: %d\n", h_out);
  if (h_out != 42)
  {
    printf("a: kernel was NOT actually launched: out != 42\n");
  }
  else
  {
    printf("a: kernel was launched: out == 42\n");
  }

  printf("\n");
}

void a()
{
  cudaGetLastError();

  size_t* d_out{};
  cudaMalloc(&d_out, sizeof(size_t));
  cudaMemset(d_out, 0, sizeof(size_t));

  void* ptr = reinterpret_cast<void*>(kernel<size_t>);

  printf("a: kernel stub address: %p\n", ptr);

  cudaFunction_t func{};
  if (cudaError_t error = cudaGetFuncBySymbol(&func, ptr))
  {
    printf("a: kernel NOT found in mapping: %s\n", cudaGetErrorString(error));
  }
  else
  {
    printf("a: kernel is in mapping: %s\n", cudaGetErrorString(error));
  }

  kernel<<<1, 1>>>('a', d_out);

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
    printf("a: FAILED to synchronize stream\n");
  }
  else
  {
    printf("a: synchronized stream\n");
  }

  size_t h_out{};
  if (cudaMemcpy(&h_out, d_out, sizeof(size_t), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    printf("a: FAILED to copy from device to host\n");
  }
  else
  {
    printf("a: copied from device to host\n");
  }

  printf("a: out: %d\n", static_cast<int>(h_out));
  if (h_out != 42)
  {
    printf("a: kernel was NOT actually launched: out != 42\n");
  }
  else
  {
    printf("a: kernel was launched: out == 42\n");
  }

  cudaMemset(d_out, 0, sizeof(size_t));
  printf("\n");

  printf("a: defers launch to b\n");
  b_launch(kernel<size_t>, 'b', d_out);
}
