#include <cub/detail/double_buffer.cuh>
#include <cub/detail/null_type.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cstddef>
#include <cstdio>

void b()
{
  printf("b() called\n");

  cub::DoubleBuffer<unsigned int> d_keys;
  cub::DoubleBuffer<cub::NullType> d_values;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d_keys, d_values, 1024);
}
