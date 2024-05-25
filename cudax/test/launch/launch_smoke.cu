#include <cuda/atomic>
#include <cuda/experimental/launch.cuh>

#include <functional>
#include <iostream>
#include <type_traits>

#include "../hierarchy/testing_common.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr unsigned int ceil(unsigned int a, unsigned int b)
{
  return (a + b - 1) / b;
}

struct my_dynamic_smem_t
{
  int arr[256];
  int i;
  float f;
};

// cudax::launch works with __global__ templates as well, it just provides less benefits
template <typename Config>
__global__ void non_functor(Config conf, int i)
{
  auto& dynamic_smem = cudax::dynamic_smem_ref(conf);

  static_assert(std::is_same_v<decltype(dynamic_smem), my_dynamic_smem_t&>);

  dynamic_smem.i = 42;
}

void non_functor_example()
{
  auto dims = cudax::block_dims<512>() & cudax::grid_dims<256>();
  auto conf = cudax::kernel_config(dims, cudax::dynamic_shared_memory<my_dynamic_smem_t>());

  auto fn = non_functor<decltype(conf)>;

  cudax::launch(conf, fn, 42U);
  cudax::launch(conf, fn, 42);
}

// cudax::launch can take functions that does not take config info as well
__global__ void old_style_kernel()
{
  if (cg::grid_group::thread_rank() == 0)
  {
    printf("Works too\n");
  }
}

void new_launch_old_kernel()
{
  auto dimensions = cudax::block_dims<512>() & cudax::grid_dims(256);

  cudax::launch(dimensions, old_style_kernel);
}

void inline_lambda_example()
{
  cudax::launch(cudax::block_dims<256>() & cudax::grid_dims(12), [] __device__(auto dims) {
    if (dims.rank(cudax::thread, cudax::block) == 0)
    {
      printf("Hello from the GPU\n");
    }
  });

  cudax::launch(cudax::block_dims<256>() & cudax::grid_dims(12), [] __device__() {});
}

// Not templated on dims for now, because it needs CG integration
__device__ void check_dyn_smem(const cg::thread_block& block, my_dynamic_smem_t& smem)
{
  if (block.thread_rank())
  {
    block.sync();
    smem.i = 42;
    block.sync();
  }
  else
  {
    smem.i = 0;
    block.sync();
    block.sync();
    assert(smem.i == 42);
  }
}

__device__ void max_out_static_smem(const cg::thread_block& block)
{
  char __shared__ array[48 * 1024];
  array[block.thread_rank()] = block.thread_rank();
  assert(array[block.thread_rank()] == block.thread_rank());
}

struct dynamic_smem_single
{
  template <typename Config>
  __device__ void operator()(Config conf)
  {
    auto block = cg::this_thread_block();
    max_out_static_smem(block);

    auto& dynamic_smem = cudax::dynamic_smem_ref(conf);
    check_dyn_smem(block, dynamic_smem);
  }
};

struct dynamic_smem_span
{
  template <typename Config>
  __device__ void operator()(Config conf)
  {
    auto block = cg::this_thread_block();
    max_out_static_smem(block);

    auto dynamic_smem = cudax::dynamic_smem_span(conf);
    check_dyn_smem(block, dynamic_smem[1]);
  }
};

void dynamic_smem_example()
{
  auto dims = cudax::block_dims<2>() & cudax::grid_dims<1>();
  auto conf = cudax::kernel_config(dims, cudax::dynamic_shared_memory<my_dynamic_smem_t>());

  cudax::launch(conf, dynamic_smem_single{});

  auto conf2 = cudax::kernel_config(dims, cudax::dynamic_shared_memory<my_dynamic_smem_t>(2));
  cudax::launch(conf2, dynamic_smem_span{});

  auto conf3 = cudax::kernel_config(dims, cudax::dynamic_shared_memory<my_dynamic_smem_t, 3>());
  cudax::launch(conf3, dynamic_smem_span{});

  auto conf_large = cudax::kernel_config(dims, cudax::dynamic_shared_memory<my_dynamic_smem_t, 48>());
  cudax::launch(conf_large, dynamic_smem_span{});
}

struct static_self_contained
{
  static constexpr auto conf =
    cudax::kernel_config(cudax::block_dims<256>() & cudax::grid_dims<128>(), cudax::dynamic_shared_memory<int>());

  __device__ void operator()(decltype(conf) config)
  {
    auto grid      = cg::this_grid();
    auto& dyn_smem = cudax::dynamic_smem_ref(config);
  }
};

void stream_example()
{
  cudaStream_t stream;

  cudaStreamCreate(&stream);

  auto dims = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(12));
  auto conf = cudax::make_config(dims, cudax::launch_on(stream));

  cudax::launch(conf, [] __device__(auto conf) {
    if (conf.dims.rank(cudax::thread) == 0)
    {
      printf("block size %d\n", blockDim.x);
    }
  });
}

TEST_CASE("Smoke", "[launch]")
{
  // Examples of use
  try
  {
    non_functor_example();
    new_launch_old_kernel();
    inline_lambda_example();
    dynamic_smem_example();
    stream_example();
  }
  catch (cuda::cuda_error& e)
  {
    printf("Launch error %s\n", e.what());
    // static_assert(cuda::std::is_base_of_v<::cuda::std::exception, cuda::cuda_error>);
    throw std::runtime_error(e.what());
  }
  cudaDeviceSynchronize();
}
