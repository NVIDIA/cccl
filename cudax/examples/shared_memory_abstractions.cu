#include <cuda/std/cassert>

#include <cuda/experimental/memory.cuh>

#include <cstdio>

namespace cudax = cuda::experimental;

#define thread_printf(FMT, ...) \
  printf("[%d, %d, %d]: " FMT, threadIdx.x, threadIdx.y, threadIdx.z __VA_OPT__(, ) __VA_ARGS__)

struct SharedObj
{
  __device__ SharedObj()
  {
    thread_printf("Default constructing...\n");
  }

  __device__ SharedObj(int v)
      : value_{v}
  {
    thread_printf("Constructing with %d...\n", v);
  }

  __device__ void use()
  {
    thread_printf("Using with value (%d)...\n", value_);
  }

  __device__ ~SharedObj()
  {
    thread_printf("Destructing...\n");
  }

  int value_{0};
};

__device__ void use(cudax::shared_memory_ptr<SharedObj> ptr)
{
  ptr->use();
}

__global__ void demo1()
{
  // The default constructor is called by thread 0.
  cudax::static_shared<SharedObj> shared_obj{};

  // Wait for the construction to complete.
  __syncthreads();

  use(&shared_obj);

  // The object will be destructed by thread 0 when the object goes out of scope.
}

__global__ void demo2()
{
  cudax::static_shared<SharedObj> shared_obj1{1};
  cudax::static_shared<SharedObj> shared_obj2{cuda::no_init};

  // Oops, is uninitialized, would cause assertion to trigger.
  // use(&shared_obj2);

  // Construct shared_obj2 by thread 1.
  shared_obj2.construct_by({1, 0, 0}, 2);

  // Wait for the construction to complete.
  __syncthreads();

  use(&shared_obj1);
  use(&shared_obj2);

  // Manually destroy the shared_obj1 by thread 0.
  shared_obj1.destroy();

  // Manually destroy the shared_obj2 by thread 1.
  shared_obj2.destroy_by({1, 0, 0});

  // Oops, is already destructed, would cause assertion to trigger.
  // shared_obj1.destroy();

  // The destructor will not do anything.
}

__global__ void demo3()
{
  // Create a shared buffer with 32 bytes of storage and 16 bytes of alignment.
  cudax::static_shared_storage<32, 16> shared_buff;

  // Obtain pointer to the shared buffer.
  auto ptr = cudax::static_pointer_cast<SharedObj>(&shared_buff);

  // Construct SharedObj by thread 0.
  if (threadIdx.x == 0)
  {
    new (ptr.get()) SharedObj{123};
  }

  // Wait for the construction to complete.
  __syncthreads();

  use(ptr);

  // Wait for all threads to complete before destructing the object.
  __syncthreads();

  // Destruct the object by thread 0.
  if (threadIdx.x == 0)
  {
    ptr->~SharedObj();
  }
}

int main()
{
  printf("Demo1:\n");
  demo1<<<1, 4>>>();
  assert(cudaDeviceSynchronize() == cudaSuccess);
  printf("\n");

  printf("Demo2:\n");
  demo2<<<1, 4>>>();
  assert(cudaDeviceSynchronize() == cudaSuccess);
  printf("\n");

  printf("Demo3:\n");
  demo3<<<1, 4>>>();
  assert(cudaDeviceSynchronize() == cudaSuccess);
  printf("\n");
}
