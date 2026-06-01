#include <cub/device/device_scan.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

#include <cstdio>

int main()
{
  thrust::device_vector<int> a{0, 2, 5, 7, 8};

  thrust::device_vector<int> b{0, 1, 2, 3, 8, 9};

  thrust::device_vector<int> out(6);
  thrust::lower_bound(a.begin(), a.end(), b.begin(), b.end(), out.begin());

  for (int a : out)
  {
    printf("%d\n", (int) a);
  }
  return 0;
}
