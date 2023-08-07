

/*
This is a simple example demonstrating the use of CCCL functionality from Thrust, CUB, and libcu++.

The example computes the sum of an array of integers using a simple parallel reduction. Each thread block
computes the sum of a subset of the array using cuB::BlockRecuce. The sum of each block is then reduced 
to a single value using an atomic add via cuda::atomic_ref from libcu++. The result is stored in a device_vector
from Thrust. The sum is then printed to the console.
*/

#include <cstdio>
#include <thrust/device_vector.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>

constexpr int block_size = 256;

__global__ void sumKernel(int const* data, int* result, std::size_t N)
{
    using BlockReduce = cub::BlockReduce<int, block_size> ;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    int sum = 0;
    if (index < N) {
        sum += data[index];
    }

    sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0){
        cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(*result);
        atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
    }
}

int main()
{
    std::size_t N = 1000;
    thrust::device_vector<int> data(N, 1);
    thrust::device_vector<int> result(1);

    int numBlocks = (N + block_size - 1) / block_size;

    sumKernel<<<numBlocks, block_size>>>(thrust::raw_pointer_cast(data.data()),
                                         thrust::raw_pointer_cast(result.data()), N);

    auto err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "Sum: " << result[0] << std::endl;

    assert(result[0] == N);

    return 0;
}
