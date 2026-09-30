> [!CAUTION]
> This is an internal tool not intended for public use.

> [!WARNING]
> This tool is experimental.

# NVRTCC

## Overview

`nvrtcc` is a tool to simplify [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html) testing. It follows the `nvcc` compilation trajectory and replaces the `nvcc` generated PTX code with the NVRTC compiled one. The main advantage is that you can keep the source files almost the same as if compiled with `nvcc`. `nvrtcc` makes sure all of the necessary symbols are present in the generated PTX and let's `nvcc` do the host compilation and linking.

The compilation with NVRTC is optional and can be controlled by the `-use-nvrtc` flag. This allows `nvrtcc` to be used as the `CMAKE_CUDA_COMPILER` where we can trick CMake to think it's using `nvcc`. `nvrtcc` supports almost all `nvcc` options except for those that are unsupported by NVRTC.

## Example

When compiling with `nvrtcc`, we need to make sure NVRTC wouldn't see the host includes and symbols, so they must be guarded by `#ifndef __CUDACC_RTC__` preprocessor directive.
```cpp
// hello_world.cu

#ifndef __CUDACC_RTC__
#  include <cstdio>
#endif

__global__ void kernel()
{
#ifdef __CUDACC_RTC__
    printf("Hello world from NVRTC!\n");
#else
    printf("Hello world from NVCC!\n");
#endif
}

#ifndef __CUDACC_RTC__
int main()
{
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
#endif
```

If we compile with `nvcc` the output should look as:
```sh
$ nvcc hello_world.cu -o hello_world
$ ./hello_world
Hello world from NVCC!
```

On the other hand if compiled with `nvrtcc` with `-use-nvrtc` flag present, the output should be:
```sh
$ nvrtcc hello_world.cu -o hello_world -use-nvrtc
$ ./hello_world
Hello world from NVRTCC!
```
