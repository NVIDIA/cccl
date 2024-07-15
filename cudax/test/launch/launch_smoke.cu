//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#define LIBCUDACXX_ENABLE_EXCEPTIONS
#include <cuda/atomic>
#include <cuda/experimental/launch.cuh>

#include <functional>
#include <iostream>
#include <type_traits>

#include "../hierarchy/testing_common.cuh"
#include <cooperative_groups.h>

__managed__ bool kernel_run_proof = false;

void check_kernel_run(cudaStream_t stream)
{
  CUDART(cudaStreamSynchronize(stream));
  CHECK(kernel_run_proof);
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
    assert(conf.dims.count(cudax::block, cudax::grid) == grid_size);
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
    assert(dims.count(cudax::block, cudax::grid) == grid_size);
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
    assert(__isShared(&dynamic_smem));
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
    assert(dynamic_smem.size() == size);
    assert(__isShared(&dynamic_smem[1]));
    kernel_run_proof = true;
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
        cudax::launch(stream, dims_or_conf, functor_int_argument(), dummy);
        check_kernel_run(stream);
        cudax::launch(stream, dims_or_conf, functor_int_argument(), 1);
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

      cudax::launch(stream, config, kernel_instance, grid_size);
      check_kernel_run(stream);
      cudax::launch(stream, config, kernel_instance, ::cuda::std::move(grid_size));
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

      cudax::launch(stream, dimensions, kernel_instance, grid_size);
      check_kernel_run(stream);
      cudax::launch(stream, dimensions, kernel_instance, ::cuda::std::move(grid_size));
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

__global__ void empty_kernel() {};

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
    auto dims = cudax::make_hierarchy(cudax::block_dims<256>(), cudax::grid_dims(cudax::at_least(1024, cudax::thread)));

    // Won't work until finalized
    // dims.count();

    // Does not touch a meta dims, so works
    static_assert(dims.count(cudax::thread, cudax::block) == 256);

    auto dims_finalized = cudax::finalize(dims, check_expected_counts);
    static_assert(::cuda::std::is_same_v<::cudax::transformed_hierarchy_t<decltype(dims)>, decltype(dims_finalized)>);

    print_dims(dims_finalized);

    CUDAX_REQUIRE(dims_finalized.count(cudax::block, cudax::grid) == 4);

    cudax::launch(stream, dims_finalized, check_expected_counts, 256, 4);

    cudax::launch(stream, dims, check_expected_counts, 256, 4);
  }

  SECTION("At least + best occupancy")
  {
    unsigned int target_count = 4420;
    auto dims                 = cudax::make_hierarchy(
      cudax::block_dims(cudax::best_occupancy()), cudax::grid_dims(cudax::at_least(target_count, cudax::thread)));

    auto dims_finalized = cudax::finalize(dims, empty_kernel);
    static_assert(::cuda::std::is_same_v<::cudax::transformed_hierarchy_t<decltype(dims)>, decltype(dims_finalized)>);

    CUDAX_REQUIRE(dims_finalized.count(cudax::thread) >= target_count);
    print_dims(dims_finalized);

    cudax::launch(stream, dims_finalized, empty_kernel);

    cudax::launch(stream, dims, empty_kernel);
  }

  SECTION("max_coresident + best occupancy")
  {
    auto dims =
      cudax::make_hierarchy(cudax::block_dims(cudax::best_occupancy()), cudax::grid_dims(cudax::max_coresident()));

    auto config = cudax::make_config(dims, cudax::cooperative_launch());

    auto config_finalized = cudax::finalize(config, grid_sync_kernel);
    static_assert(::cuda::std::is_same_v<::cudax::transformed_config_t<decltype(config)>, decltype(config_finalized)>);

    print_dims(config_finalized.dims);

    cudax::launch(stream, config_finalized, grid_sync_kernel, 1);

    auto config_finalized_with_arguments = cudax::finalize(config, grid_sync_kernel, 1);
    static_assert(::cuda::std::is_same_v<decltype(config_finalized_with_arguments), decltype(config_finalized)>);

    cudax::launch(stream, config_finalized_with_arguments, grid_sync_kernel, 1);

    cudax::launch(stream, config, grid_sync_kernel, 1);

    auto lambda = [] __device__(auto dims, int dummy) {
      auto grid = cooperative_groups::this_grid();
      grid.sync();
    };

    auto finalized_for_lambda = cudax::finalize(config, lambda, 1);
    static_assert(
      ::cuda::std::is_same_v<::cudax::transformed_config_t<decltype(config)>, decltype(finalized_for_lambda)>);

    print_dims(finalized_for_lambda.dims);

    cudax::launch(stream, finalized_for_lambda, lambda, 1);

    cudax::launch(stream, config, lambda, 1);
  }

  SECTION("Dyn smem and max coresident")
  {
    constexpr unsigned int large_smem = 7 * 1024;
    auto dims = cudax::make_hierarchy(cudax::block_dims(128), cudax::grid_dims(cudax::max_coresident()));

    auto dims_transformed = cudax::finalize(dims, shared_memory_expected_counts<>);
    print_dims(dims_transformed);

    auto dims_with_smem = cudax::finalize(dims, shared_memory_expected_counts<large_smem>);
    print_dims(dims_with_smem);

    auto config = cudax::make_config(dims, cudax::dynamic_shared_memory<int>(large_smem));

    auto config_transformed = cudax::finalize(config, shared_memory_expected_counts<>);
    print_dims(config_transformed.dims);

    CUDAX_REQUIRE(dims_transformed.count(cudax::thread, cudax::block)
                  == config_transformed.dims.count(cudax::thread, cudax::block));
    CUDAX_REQUIRE(
      dims_transformed.count(cudax::thread, cudax::block) == dims_with_smem.count(cudax::thread, cudax::block));

    // Confirm adding large dynamic shared reduced the number of blocks that can execute at the same time
    CUDAX_REQUIRE(dims_transformed.count(cudax::block) > config_transformed.dims.count(cudax::block));
    // But the number is the same no matter if smem is static or dynamic
    CUDAX_REQUIRE(dims_with_smem.count(cudax::block) == config_transformed.dims.count(cudax::block));

    cudax::launch(stream,
                  dims,
                  shared_memory_expected_counts<>,
                  dims_transformed.count(cudax::thread, cudax::block),
                  dims_transformed.count(cudax::block));

    cudax::launch(stream,
                  dims,
                  shared_memory_expected_counts<large_smem>,
                  dims_with_smem.count(cudax::thread, cudax::block),
                  dims_with_smem.count(cudax::block));

    cudax::launch(stream,
                  config,
                  shared_memory_expected_counts<>,
                  config_transformed.dims.count(cudax::thread, cudax::block),
                  config_transformed.dims.count(cudax::block));
  }
  CUDART(cudaStreamSynchronize(stream));
  CUDART(cudaStreamDestroy(stream));
}

TEST_CASE("Meta dimensions", "[launch]")
{
  meta_dims_test();
}
