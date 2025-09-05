#include "kernel.cuh"

void a_launch(void (*k)(char, int*), char c, int* d_out);

void b_launch(void (*k)(char, size_t*), char c, size_t* d_out)
{
  void* ptr = reinterpret_cast<void*>(k);

  std::printf("b: kernel stub address: %p\n", ptr);

  cudaFunction_t func{};
  if (cudaError_t error = cudaGetFuncBySymbol(&func, ptr))
  {
    std::printf("b: kernel NOT found in mapping: %s\n", cudaGetErrorString(error));
  }
  else
  {
    std::printf("b: kernel is in mapping: %s\n", cudaGetErrorString(error));
  }

  k<<<1, 1>>>(c, d_out);

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
    std::printf("b: FAILED to synchronize stream\n");
  }
  else
  {
    std::printf("b: synchronized stream\n");
  }

  size_t h_out{};
  if (cudaMemcpy(&h_out, d_out, sizeof(size_t), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    std::printf("b: FAILED to copy from device to host\n");
  }
  else
  {
    std::printf("b: copied from device to host\n");
  }

  std::printf("b: out: %d\n", static_cast<int>(h_out));
  if (h_out != 42)
  {
    std::printf("b: kernel was NOT actually launched: out != 42\n");
  }
  else
  {
    std::printf("b: kernel was launched: out == 42\n");
  }

  std::printf("\n");
}

void b()
{
  cudaGetLastError();

  int* d_out{};
  cudaMalloc(&d_out, sizeof(int));
  cudaMemset(d_out, 0, sizeof(int));

  void* ptr = reinterpret_cast<void*>(kernel<int>);

  std::printf("b: kernel stub address: %p\n", ptr);

  cudaFunction_t func{};
  if (cudaError_t error = cudaGetFuncBySymbol(&func, ptr))
  {
    std::printf("b: kernel NOT found in mapping: %s\n", cudaGetErrorString(error));
  }
  else
  {
    std::printf("b: kernel is in mapping: %s\n", cudaGetErrorString(error));
  }

  kernel<<<1, 1>>>('b', d_out);

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
    std::printf("b: FAILED to synchronize stream\n");
  }
  else
  {
    std::printf("b: synchronized stream\n");
  }

  int h_out{};
  if (cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    std::printf("b: FAILED to copy from device to host\n");
  }
  else
  {
    std::printf("b: copied from device to host\n");
  }

  std::printf("b: out: %d\n", h_out);

  if (h_out != 42)
  {
    std::printf("b: kernel was NOT actually launched: out != 42\n");
  }
  else
  {
    std::printf("b: kernel was launched: out == 42\n");
  }

  cudaMemset(d_out, 0, sizeof(int));
  std::printf("\n");

  std::printf("b: defers launch to a\n");
  a_launch(kernel<int>, 'b', d_out);
}
