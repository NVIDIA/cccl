#include <thrust/device_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/multi_device_resource.h>
#include <thrust/mr/polymorphic_adaptor.h>
#include <thrust/mr/new.h>
#include <thrust/mr/pool.h>
#include <thrust/mr/disjoint_pool.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

// Simple CUDA error checker
#define CHECK_CUDA(call)                                                   \
  {                                                                        \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                << " code=" << err << " \"" << cudaGetErrorString(err)     \
                << "\"" << std::endl;                                      \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }

// Kernel for simple computation
__global__ void add_kernel(int* data, int n, int value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += value;
}

// Generic vector test 
template <typename Vec>
void test_vector(typename Vec::allocator_type alloc)
{
    Vec v1(alloc);
    v1.push_back(1);
    assert(v1.back() == 1);

    Vec v2(alloc);
    v2 = v1;
    v1.swap(v2);

    v1.clear();
    v1.resize(2);
    assert(v1.size() == 2);
}

// Multi-device test for a GPU vector
void test_multi_device_vector(
    int device_id,
    thrust::mr::multi_device_async_resource<thrust::device_ptr<void>>* multi_res)
{
    CHECK_CUDA(cudaSetDevice(device_id));

    // Polymorphic allocator per device
    thrust::mr::polymorphic_adaptor_resource<thrust::device_ptr<void>> adaptor(multi_res);
    using Alloc = thrust::mr::polymorphic_allocator<int, thrust::device_ptr<void>>;
    Alloc alloc(&adaptor);

    thrust::device_vector<int, Alloc> vec(10, 1, alloc);

    int* raw_ptr = thrust::raw_pointer_cast(vec.data());
    int n = vec.size();
    int blockSize = 128;
    int gridSize = (n + blockSize - 1) / blockSize;
    add_kernel<<<gridSize, blockSize>>>(raw_ptr, n, 42);
    CHECK_CUDA(cudaDeviceSynchronize());

    thrust::host_vector<int> host_vec = vec;
    std::cout << "Device " << device_id << " vector after add_kernel: ";
    for (int i = 0; i < host_vec.size(); ++i) std::cout << host_vec[i] << " ";
    std::cout << std::endl;
}

int main()
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "Requires at least 2 CUDA devices.\n";
        return 0;
    }

    // Underlying resource
    thrust::mr::cuda_memory_resource cuda_res;

    // Multi-device async resource (manages allocations per device/stream)
    thrust::mr::multi_device_async_resource<thrust::device_ptr<void>> multi_res(&cuda_res);

    // ----- Host-style examples (original idea) -----
    thrust::mr::new_delete_resource host_res;

    {
        using Alloc = thrust::mr::allocator<int, thrust::mr::new_delete_resource>;
        Alloc alloc(&host_res);
        test_vector<thrust::host_vector<int, Alloc>>(alloc);
    }

    {
        thrust::mr::polymorphic_adaptor_resource<void*> adaptor(&host_res);
        using Alloc = thrust::mr::polymorphic_allocator<int, void*>;
        Alloc alloc(&adaptor);
        test_vector<thrust::host_vector<int, Alloc>>(alloc);
    }

    {
        using Pool = thrust::mr::unsynchronized_pool_resource<thrust::mr::new_delete_resource>;
        Pool pool(&host_res);
        using Alloc = thrust::mr::allocator<int, Pool>;
        Alloc alloc(&pool);
        test_vector<thrust::host_vector<int, Alloc>>(alloc);
    }

    {
        using DisjointPool =
            thrust::mr::disjoint_unsynchronized_pool_resource<thrust::mr::new_delete_resource,
                                                              thrust::mr::new_delete_resource>;
        DisjointPool disjoint_pool(&host_res, &host_res);
        using Alloc = thrust::mr::allocator<int, DisjointPool>;
        Alloc alloc(&disjoint_pool);
        test_vector<thrust::host_vector<int, Alloc>>(alloc);
    }

    // ----- Multi-device GPU examples -----
    int max_devices = std::min(device_count, 4);
    for (int dev = 0; dev < max_devices; ++dev) {
        test_multi_device_vector(dev, &multi_res);
    }

    std::cout << "Multi-device async buffer + host vector demo completed successfully.\n";
    return 0;
}
