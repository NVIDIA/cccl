#include <cub/cub.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/experimental/memory_resource.cuh>
#include <cuda/experimental/stream.cuh>

#include <iostream>
#include <vector>

namespace cudax = cuda::experimental;

// Compile with nvcc  -Icub -Ilibcudacxx/include -Icudax/include -Ithrust/ -Icudax/include/ sample_device_segmented_reduce_env.cu -o sample_device_segmented_reduce_env

// In cccl/ repo

int main()
{

  int num_segments = 3;

  
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  int* d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());

  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  int* d_in_it = thrust::raw_pointer_cast(d_in.data());

  thrust::device_vector<int> d_out(num_segments);
  int* d_out_it = thrust::raw_pointer_cast(d_out.data());

  // Build an env (from https://github.com/NVIDIA/cccl/blob/main/cudax/examples/cub_reduce.cu)
  // A CUDA stream on which to execute the reduction
  cuda::stream stream{cuda::devices[0]};
  cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(cuda::devices[0]);

  // An environment we use to pass all necessary information to CUB
  cudax::env_t<cuda::mr::device_accessible> env{mr, stream};

  cub::DeviceSegmentedReduce::Sum(d_in_it, d_out_it, num_segments, d_offsets_it, d_offsets_it + 1, env);

  thrust::host_vector<int> h_out = d_out;
  thrust::host_vector<int> expected{21, 0, 17};

  std::cout << "Segmented reduce results:\n";
  for (int i = 0; i < num_segments; ++i)
  {
    std::cout << "Segment " << i << ": " << h_out[i] << " (expected " << expected[i] << ")\n";
  }

  return 0;
}
