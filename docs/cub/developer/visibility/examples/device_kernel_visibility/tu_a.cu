#include "kernel.cuh"

void b_launch(void (*k)(char, size_t*), char c, size_t* d_out);

void a_launch(void (*k)(char, int*), char c, int* d_out)
{
  void* ptr = reinterpret_cast<void*>(k);

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

  cudaMemset(d_out, 0, sizeof(int));
  k<<<1, 1>>>(c, d_out);

  if (cudaPeekAtLastError() != cudaSuccess)
  {
    std::printf("a: FAILED to launch kernel\n");
  }
  else
  {
    std::printf("a: launched kernel\n");
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

void a()
{
  cudaGetLastError();

  size_t* d_out{};
  cudaMalloc(&d_out, sizeof(size_t));
  cudaMemset(d_out, 0, sizeof(size_t));

  void* ptr = reinterpret_cast<void*>(kernel<size_t>);

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

  size_t h_out{};
  if (cudaMemcpy(&h_out, d_out, sizeof(size_t), cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    std::printf("a: FAILED to copy from device to host\n");
  }
  else
  {
    std::printf("a: copied from device to host\n");
  }

  std::printf("a: out: %d\n", static_cast<int>(h_out));
  if (h_out != 42)
  {
    std::printf("a: kernel was NOT actually launched: out != 42\n");
  }
  else
  {
    std::printf("a: kernel was launched: out == 42\n");
  }

  cudaMemset(d_out, 0, sizeof(size_t));
  std::printf("\n");

  std::printf("a: defers launch to b\n");
  b_launch(kernel<size_t>, 'b', d_out);
}
