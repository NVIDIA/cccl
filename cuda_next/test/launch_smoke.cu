#include <cuda/atomic>
#include <cuda/next/launch.cuh>

#include <functional>
#include <iostream>
#include <type_traits>

#include "catch2_helpers/testing_common.cuh"
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

/*
struct self_contained {
    unsigned int problem_size;

    template <Arch>
    auto get_config(unsigned int priority, cudaStream_t stream) {
        auto constexpr block_size = 256;
        auto dims = cudax::block<map(arch -> unsigned int)>() & cudax::grid(problem_size / block_size);
        auto config = cudax::cooperative_launch() & cudax::launch_priority(priority) & cudax::arch<Arch> &
cudax::launch_stream(stream);

        return cudax::kernel_config(dims, config);
    }

    // Technically could be deduced from the above, but this is simpler
    template <typename Config>
    __device__ void operator()(Config conf, unsigned int i) {

    }
};

void self_contained_example() {
    cudax::launch(static_self_contained::conf, static_self_contained{});

    self_contained kernel = {2048};
    cudax::launch(kernel.get_config(3), kernel, 2U);
}
*/

// Non-functional mock-up of possible future features accessible with cudax::launch
// It shows a possible API to attach small amount of global memory to a kernel and to a one-way or two-way copy to it
// Its both better performing than explicit cudaMemcpy and easier to use
struct cooperative_counter
{
  template <typename Configuration, typename T>
  __device__ void operator()(Configuration config, T* data, size_t size, T searched)
  {
    auto& atomic_int = get_workspace_content(config);

    for (size_t i = 0; i < size; i += cudax::grid.count(cudax::thread))
    {
      if (data[i + cudax::grid.rank(cudax::thread)] == searched)
      {
        atomic_int += 1;
      }
    }
  }
};

/*
void workspace_example()
{
  cuda::atomic<int> atomic_int(0);
  auto dimensions = cudax::block_dims<256>() & cudax::cluster_dims<2, 2, 2>() & cudax::grid_dims<128>();
  auto extensions = cudax::workspace(atomic_int) & cudax::cooperative_launch();

  thrust::device_vector<int> vec(1024);
  thrust::sequence(vec.begin(), vec.end());

  // Workspace is initialized with contents of atomic_int variable as part of launch,
  // making it faster than a separate memcpy for a small object (important for short-running kernels)
  cudax::launch(dimensions & extensions, cooperative_counter{}, thrust::raw_pointer_cast(vec.data()), vec.size(), 42);
  cudaDeviceSynchronize();
  // Value from the workspace is copied back after kernel exits (with a normal memcpy this time)
  // Can be opted out, if not needed

  std::cout << atomic_int << std::endl;
}*/

void per_arch_example(int i)
{
  static auto configs = cudax::per_arch_kernel_config(
    cudax::sm_60() >>= cudax::block_dims<256>() & cudax::grid_dims(i) & cudax::cooperative_launch(),
    cudax::sm_70() >>= (cudax::block_dims<512>() & cudax::grid_dims<128>() & cudax::cooperative_launch()),
    cudax::arch_specific_config(
      cudax::sm_80(), cudax::block_dims<64>() & cudax::grid_dims<128>() & cudax::cooperative_launch()),
    cudax::arch_specific_config(
      cudax::sm_90(), cudax::block_dims<1024>() & cudax::grid_dims<128>() & cudax::cooperative_launch()));
  static constexpr auto& c = cudax::get_target_config<cudax::sm_70>(configs);
  static_assert(c.dims.count() == 512 * 128);
}

/*
struct kernel
{
  template <typename Config>
  __device__ void operator()(Config config)
  {
    static_assert(cudax::is_cooperative(config));
    auto& dynamic_smem = cudax::get_smem_content_ref(config);
  }
};

template <typename NonDeduced>
struct kernel
{
  int param1;
  int param2;
  int param3;

  auto get_config()
  {
    return cudax::kernel_config();
  }

  using config_type = // can't make it work

    __device__ void operator()(config_type config, int param4)
  {
    static_assert(cudax::is_cooperative(config));
    auto& dynamic_smem = cudax::get_smem_content_ref(config);
  }
};

auto k = kernel<int>(p1, p2, p3);
cudax::launch(k.get_config(), k, p4);
cudax::launch(k, p4);
*/

void stream_example()
{
  cudaStream_t stream;

  cudaStreamCreate(&stream);

  auto dims = cudax::make_hierarchy(cudax::block_dims<128>(), cudax::grid_dims(12));
  auto conf = cudax::make_config(dims, cudax::launch_on(stream));

  cudax::launch(conf, [] __device__() {});
}

TEST_CASE("Smoke", "[launch]")
{
  // Examples of use
  try
  {
    // simple_reduce_example();
    non_functor_example();
    new_launch_old_kernel();
    inline_lambda_example();
    dynamic_smem_example();
    per_arch_example(1);
    stream_example();
    //        self_contained_example();
  }
  catch (cudax::launchErrorException e)
  {
    printf("Launch error %d\n", e.error);
  }
  cudaDeviceSynchronize();
  // Pure mock-up, would require more work to implement
  // workspace_example();
}
