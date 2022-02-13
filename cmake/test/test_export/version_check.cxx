#include <cuda/std/atomic>

#include <cstdio>

int main()
{
  cuda::std::atomic<int> x{0};

  printf("Built with libcudacxx version %d.%d.%d.\n",
         _LIBCUDACXX_CUDA_API_VERSION_MAJOR,
         _LIBCUDACXX_CUDA_API_VERSION_MINOR,
         _LIBCUDACXX_CUDA_API_VERSION_PATCH);

  return x;
}
