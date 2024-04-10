#include <stdio.h>

#define CUDA_CALL(...)                                                                \
  do                                                                                  \
  {                                                                                   \
    cudaError_t err = __VA_ARGS__;                                                    \
    if (err != cudaSuccess)                                                           \
    {                                                                                 \
      printf("CUDA ERROR: %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
      return err;                                                                     \
    }                                                                                 \
  } while (false)

int main()
{
  int selected_device;
  CUDA_CALL(cudaGetDevice(&selected_device));

  cudaDeviceProp device_prop;
  CUDA_CALL(cudaGetDeviceProperties(&device_prop, selected_device));

  FILE* output = fopen("sm", "w");
  fprintf(output, "%d%d\n", device_prop.major, device_prop.minor);
  fclose(output);
}
