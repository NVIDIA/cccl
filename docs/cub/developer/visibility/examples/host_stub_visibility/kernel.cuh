#pragma once

#include <cstdio>

template <class T>
__global__ void kernel(char ln, T* val)
{
  std::printf("%c: kernel: set val = 42\n", ln);
  *val = 42;
}
