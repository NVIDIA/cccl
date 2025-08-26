#include <cuda/experimental/cufile.h>

#include <iostream>

#include <cuda_runtime.h>

int main()
{
#if CUDAX_HAS_CUFILE()
  try
  {
    cuda::experimental::cufile::driver_handle driver; // RAII driver open/close

    constexpr size_t size = 1 << 20;
    void* dev_ptr         = nullptr;
    if (cudaMalloc(&dev_ptr, size) != cudaSuccess)
    {
      std::cerr << "cudaMalloc failed\n";
      return 1;
    }

    // Write example
    {
      cuda::experimental::cufile::file_handle fh{"cufile_example.bin", std::ios_base::out};
      cuda::std::span<const char> out_span{static_cast<const char*>(dev_ptr), size};
      fh.write(out_span);
    }

    // Read example
    {
      cuda::experimental::cufile::file_handle fh{"cufile_example.bin", std::ios_base::in};
      cuda::std::span<char> in_span{static_cast<char*>(dev_ptr), size};
      fh.read(in_span);
    }

    cudaFree(dev_ptr);
    std::cout << "cuFILE basic example completed\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
#else
  std::cout << "CUDAX_HAS_CUFILE=0: cuFILE not available on this system\n";
  return 0;
#endif
}
