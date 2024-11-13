//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/atomic>

#include <cuda/experimental/launch.cuh>

#include <cooperative_groups.h>
#include <host_device.cuh>

__managed__ bool kernel_run_proof = false;

void check_kernel_run(cudaStream_t stream)
{
  CUDART(cudaStreamSynchronize(stream));
  CUDAX_CHECK(kernel_run_proof);
  kernel_run_proof = false;
}

struct functor_int_argument
{
  __device__ void operator()(int dummy)
  {
    kernel_run_proof = true;
  }
};

template <unsigned int BlockSize>
struct functor_taking_config
{
  template <typename Config>
  __device__ void operator()(Config conf, int grid_size)
  {
    static_assert(conf.dims.static_count(cudax::thread, cudax::block) == BlockSize);
    CUDAX_REQUIRE(conf.dims.count(cudax::block, cudax::grid) == grid_size);
    kernel_run_proof = true;
  }
};

template <unsigned int BlockSize>
struct functor_taking_dims
{
  template <typename Dimensions>
  __device__ void operator()(Dimensions dims, int grid_size)
  {
    static_assert(dims.static_count(cudax::thread, cudax::block) == BlockSize);
    CUDAX_REQUIRE(dims.count(cudax::block, cudax::grid) == grid_size);
    kernel_run_proof = true;
  }
};

__global__ void kernel_int_argument(int dummy)
{
  kernel_run_proof = true;
}

template <typename Config, unsigned int BlockSize>
__global__ void kernel_taking_config(Config conf, int grid_size)
{
  functor_taking_config<BlockSize>()(conf, grid_size);
}

template <typename Dims, unsigned int BlockSize>
__global__ void kernel_taking_dims(Dims dims, int grid_size)
{
  functor_taking_dims<BlockSize>()(dims, grid_size);
}

struct my_dynamic_smem_t
{
  int i;
};

template <typename SmemType>
struct dynamic_smem_single
{
  template <typename Config>
  __device__ void operator()(Config conf)
  {
    auto& dynamic_smem = cudax::dynamic_smem_ref(conf);
    static_assert(::cuda::std::is_same_v<SmemType&, decltype(dynamic_smem)>);
    CUDAX_REQUIRE(__isShared(&dynamic_smem));
    kernel_run_proof = true;
  }
};

template <typename SmemType, size_t Extent>
struct dynamic_smem_span
{
  template <typename Config>
  __device__ void operator()(Config conf, int size)
  {
    auto dynamic_smem = cudax::dynamic_smem_span(conf);
    static_assert(decltype(dynamic_smem)::extent == Extent);
    static_assert(::cuda::std::is_same_v<SmemType&, decltype(dynamic_smem[1])>);
    CUDAX_REQUIRE(dynamic_smem.size() == size);
    CUDAX_REQUIRE(__isShared(&dynamic_smem[1]));
    kernel_run_proof = true;
  }
};

struct launch_transform_to_int_convertible
{
  int value_;

  struct int_convertible
  {
    cudaStream_t stream_;
    int value_;

    int_convertible(cudaStream_t stream, int value) noexcept
        : stream_(stream)
        , value_(value)
    {
      // Check that the constructor runs before the kernel is launched
      CUDAX_CHECK_FALSE(kernel_run_proof);
    }

    // Immovable to ensure that __launch_transform doesn't copy the returned
    // object
    int_convertible(int_convertible&&) = delete;

    ~int_convertible() noexcept
    {
      // Check that the destructor runs after the kernel is launched
      CUDART(cudaStreamSynchronize(stream_));
      CUDAX_CHECK(kernel_run_proof);
    }

    using __as_kernel_arg = int;

    // This is the value that will be passed to the kernel
    explicit operator int() const
    {
      return value_;
    }
  };

  _CCCL_NODISCARD_FRIEND int_convertible
  __cudax_launch_transform(::cuda::stream_ref stream, launch_transform_to_int_convertible self) noexcept
  {
    return int_convertible(stream.get(), self.value_);
  }
};

// Needs a separe function for Windows extended lambda
void launch_smoke_test()
{
  // Use raw stream to make sure it can be implicitly converted on call to launch
  cudaStream_t stream;

  CUDART(cudaStreamCreate(&stream));
  // Spell out all overloads to make sure they compile, include a check for implicit conversions
  SECTION("Launch overloads")
  {
    const int grid_size      = 4;
    constexpr int block_size = 256;
    auto dimensions          = cudax::make_hierarchy(cudax::grid_dims(grid_size), cudax::block_dims<256>());
    auto config              = cudax::make_config(dimensions);

    SECTION("Not taking dims")
    {
      auto lambda = [&](auto dims_or_conf) {
        const int dummy = 1;
        cudax::launch(stream, dims_or_conf, kernel_int_argument, dummy);
        check_kernel_run(stream);
        cudax::launch(stream, dims_or_conf, kernel_int_argument, 1);
        check_kernel_run(stream);
        cudax::launch(stream, dims_or_conf, kernel_int_argument, launch_transform_to_int_convertible{1});
        check_kernel_run(stream);
        cudax::launch(stream, dims_or_conf, functor_int_argument(), dummy);
        check_kernel_run(stream);
        cudax::launch(stream, dims_or_conf, functor_int_argument(), 1);
        check_kernel_run(stream);
        cudax::launch(stream, dims_or_conf, functor_int_argument(), launch_transform_to_int_convertible{1});
        check_kernel_run(stream);

        cudax::launch(stream, dims_or_conf, kernel_int_argument, 1U);
        check_kernel_run(stream);
        cudax::launch(stream, dims_or_conf, functor_int_argument(), 1U);
        check_kernel_run(stream);
      };
      lambda(config);
      lambda(dimensions);
    }

    SECTION("Config argument")
    {
      auto functor_instance = functor_taking_config<block_size>();
      auto kernel_instance  = kernel_taking_config<decltype(config), block_size>;

      cudax::launch(stream, config, functor_instance, grid_size);
      check_kernel_run(stream);
      cudax::launch(stream, config, functor_instance, ::cuda::std::move(grid_size));
      check_kernel_run(stream);
      cudax::launch(stream, config, functor_instance, launch_transform_to_int_convertible{grid_size});
      check_kernel_run(stream);

      cudax::launch(stream, config, kernel_instance, grid_size);
      check_kernel_run(stream);
      cudax::launch(stream, config, kernel_instance, ::cuda::std::move(grid_size));
      check_kernel_run(stream);
      cudax::launch(stream, config, kernel_instance, launch_transform_to_int_convertible{grid_size});
      check_kernel_run(stream);

      cudax::launch(stream, config, functor_instance, static_cast<unsigned int>(grid_size));
      check_kernel_run(stream);
      cudax::launch(stream, config, kernel_instance, static_cast<unsigned int>(grid_size));
      check_kernel_run(stream);
    }

    SECTION("Dimensions argument")
    {
      auto functor_instance = functor_taking_dims<block_size>();
      auto kernel_instance  = kernel_taking_dims<decltype(dimensions), block_size>;

      cudax::launch(stream, dimensions, functor_instance, grid_size);
      check_kernel_run(stream);
      cudax::launch(stream, dimensions, functor_instance, ::cuda::std::move(grid_size));
      check_kernel_run(stream);
      cudax::launch(stream, dimensions, functor_instance, launch_transform_to_int_convertible{grid_size});
      check_kernel_run(stream);

      cudax::launch(stream, dimensions, kernel_instance, grid_size);
      check_kernel_run(stream);
      cudax::launch(stream, dimensions, kernel_instance, ::cuda::std::move(grid_size));
      check_kernel_run(stream);
      cudax::launch(stream, dimensions, kernel_instance, launch_transform_to_int_convertible{grid_size});
      check_kernel_run(stream);

      cudax::launch(stream, dimensions, functor_instance, static_cast<unsigned int>(grid_size));
      check_kernel_run(stream);
      cudax::launch(stream, dimensions, kernel_instance, static_cast<unsigned int>(grid_size));
      check_kernel_run(stream);
    }
  }

  SECTION("Lambda")
  {
    cudax::launch(stream, cudax::block_dims<256>() & cudax::grid_dims(1), [] __device__(auto dims) {
      if (dims.rank(cudax::thread, cudax::block) == 0)
      {
        printf("Hello from the GPU\n");
        kernel_run_proof = true;
      }
    });
    check_kernel_run(stream);
  }

  SECTION("Dynamic shared memory option")
  {
    auto dims   = cudax::block_dims<32>() & cudax::grid_dims<1>();
    auto config = cudax::kernel_config(dims);

    auto test = [stream](const auto& input_config) {
      SECTION("Single element")
      {
        auto config = input_config.add(cudax::dynamic_shared_memory<my_dynamic_smem_t>());

        cudax::launch(stream, config, dynamic_smem_single<my_dynamic_smem_t>());
        check_kernel_run(stream);
      }

      SECTION("Dynamic span")
      {
        const int size = 2;
        auto config    = input_config.add(cudax::dynamic_shared_memory<my_dynamic_smem_t>(size));
        cudax::launch(stream, config, dynamic_smem_span<my_dynamic_smem_t, ::cuda::std::dynamic_extent>(), size);
        check_kernel_run(stream);
      }

      SECTION("Static span")
      {
        constexpr int size = 3;
        auto config        = input_config.add(cudax::dynamic_shared_memory<my_dynamic_smem_t, size>());
        cudax::launch(stream, config, dynamic_smem_span<my_dynamic_smem_t, size>(), size);
        check_kernel_run(stream);
      }
    };

    test(config);
    test(config.add(cudax::cooperative_launch(), cudax::launch_priority(0)));
  }

  CUDART(cudaStreamSynchronize(stream));
  CUDART(cudaStreamDestroy(stream));
}

TEST_CASE("Smoke", "[launch]")
{
  launch_smoke_test();
}

__global__ void check_expected_counts(unsigned int num_threads_in_block, unsigned int num_blocks_in_grid)
{
  CUDAX_REQUIRE(cudax::block.count(cudax::thread) == num_threads_in_block);
  CUDAX_REQUIRE(cudax::grid.count(cudax::block) == num_blocks_in_grid);
};

template <unsigned int Count = 0>
__global__ void shared_memory_expected_counts(unsigned int num_threads_in_block, unsigned int num_blocks_in_grid)
{
  CUDAX_REQUIRE(cudax::block.count(cudax::thread) == num_threads_in_block);
  CUDAX_REQUIRE(cudax::grid.count(cudax::block) == num_blocks_in_grid);

  if constexpr (Count != 0)
  {
    __shared__ int smem[Count];
    smem[threadIdx.x] = 1;
    CUDAX_REQUIRE(smem[threadIdx.x] == 1);
  }
}

__global__ void grid_sync_kernel(int i)
{
  auto grid = cooperative_groups::this_grid();
  grid.sync();
};

template <typename Dims>
inline void print_dims(const Dims& in)
{
  std::cout << in.count() << " block: " << in.count(cudax::thread, cudax::block) << " grid: " << in.count(cudax::block)
            << std::endl;
}

void meta_dims_test()
{
  cudaStream_t stream;
  CUDART(cudaStreamCreate(&stream));

  SECTION("Just at least")
  {
    constexpr unsigned int block_size = 256, grid_size = 4;
    auto dims = cudax::make_hierarchy(
      cudax::block_dims<block_size>(), cudax::grid_dims(cudax::at_least(block_size * grid_size, cudax::thread)));

    // Won't work until finalized
    // dims.count();

    // Does not touch a meta dims, so works
    static_assert(dims.count(cudax::thread, cudax::block) == block_size);

    auto dims_finalized = cudax::finalize(stream, dims, check_expected_counts);
    static_assert(::cuda::std::is_same_v<::cudax::finalized_t<decltype(dims)>, decltype(dims_finalized)>);

    CUDAX_REQUIRE(dims_finalized.count(cudax::block, cudax::grid) == grid_size);

    cudax::launch(stream, dims_finalized, check_expected_counts, block_size, grid_size);

    cudax::launch(stream, dims, check_expected_counts, block_size, grid_size);
  }

  SECTION("At least with adjacent level")
  {
    // Not the best usage, but should work too
    constexpr unsigned int block_size = 256, grid_size = 4;
    auto dims = cudax::make_hierarchy(
      cudax::block_dims<block_size>(), cudax::grid_dims(cudax::at_least(grid_size, cudax::block)));

    cudax::launch(stream, dims, check_expected_counts, block_size, grid_size);
  }

  SECTION("At least + best occupancy")
  {
    unsigned int target_count = 4420;
    auto dims                 = cudax::make_hierarchy(
      cudax::block_dims(cudax::max_occupancy()), cudax::grid_dims(cudax::at_least(target_count, cudax::thread)));

    auto dims_finalized = cudax::finalize(stream, dims, empty_kernel);
    static_assert(::cuda::std::is_same_v<::cudax::finalized_t<decltype(dims)>, decltype(dims_finalized)>);

    CUDAX_REQUIRE(dims_finalized.count(cudax::thread) >= target_count);

    cudax::launch(stream, dims_finalized, empty_kernel, 1);

    cudax::launch(stream, dims, empty_kernel, 1);
  }

  SECTION("max_coresident + best occupancy")
  {
    auto dims =
      cudax::make_hierarchy(cudax::block_dims(cudax::max_occupancy()), cudax::grid_dims(cudax::max_coresident()));

    auto config = cudax::make_config(dims, cudax::cooperative_launch());

    auto config_finalized = cudax::finalize(stream, config, grid_sync_kernel);
    static_assert(::cuda::std::is_same_v<::cudax::finalized_t<decltype(config)>, decltype(config_finalized)>);

    cudax::launch(stream, config_finalized, grid_sync_kernel, 1);

    auto config_finalized_with_arguments = cudax::finalize(stream, config, grid_sync_kernel, 1);
    static_assert(::cuda::std::is_same_v<decltype(config_finalized_with_arguments), decltype(config_finalized)>);

    cudax::launch(stream, config_finalized_with_arguments, grid_sync_kernel, 1);

    cudax::launch(stream, config, grid_sync_kernel, 1);

    auto lambda = [] __device__(auto conf, int dummy) {
      auto grid = cooperative_groups::this_grid();
      grid.sync();
      conf.dims.count();
    };

    auto finalized_for_lambda = cudax::finalize(stream, config, lambda, 1);
    static_assert(::cuda::std::is_same_v<::cudax::finalized_t<decltype(config)>, decltype(finalized_for_lambda)>);

    cudax::launch(stream, finalized_for_lambda, lambda, 1);

    cudax::launch(stream, config, lambda, 1);
  }

  SECTION("Dyn smem and max coresident")
  {
    constexpr unsigned int large_smem = 7 * 1024;
    auto dims = cudax::make_hierarchy(cudax::block_dims(128), cudax::grid_dims(cudax::max_coresident()));

    auto dims_finalized = cudax::finalize(stream, dims, shared_memory_expected_counts<>);

    auto dims_with_smem = cudax::finalize(stream, dims, shared_memory_expected_counts<large_smem>);

    auto config = cudax::make_config(dims, cudax::dynamic_shared_memory<int>(large_smem));

    auto config_finalized = cudax::finalize(stream, config, shared_memory_expected_counts<>);

    CUDAX_REQUIRE(
      dims_finalized.count(cudax::thread, cudax::block) == config_finalized.dims.count(cudax::thread, cudax::block));
    CUDAX_REQUIRE(
      dims_finalized.count(cudax::thread, cudax::block) == dims_with_smem.count(cudax::thread, cudax::block));

    // Confirm adding large dynamic shared reduced the number of blocks that can execute at the same time
    CUDAX_REQUIRE(dims_finalized.count(cudax::block) > config_finalized.dims.count(cudax::block));
    // But the number is the same no matter if smem is static or dynamic
    CUDAX_REQUIRE(dims_with_smem.count(cudax::block) == config_finalized.dims.count(cudax::block));

    cudax::launch(stream,
                  dims,
                  shared_memory_expected_counts<>,
                  dims_finalized.count(cudax::thread, cudax::block),
                  dims_finalized.count(cudax::block));

    cudax::launch(stream,
                  dims,
                  shared_memory_expected_counts<large_smem>,
                  dims_with_smem.count(cudax::thread, cudax::block),
                  dims_with_smem.count(cudax::block));

    cudax::launch(stream,
                  config,
                  shared_memory_expected_counts<>,
                  config_finalized.dims.count(cudax::thread, cudax::block),
                  config_finalized.dims.count(cudax::block));
  }
  CUDART(cudaStreamSynchronize(stream));
  CUDART(cudaStreamDestroy(stream));
}

TEST_CASE("Meta dimensions", "[launch]")
{
  meta_dims_test();
}
