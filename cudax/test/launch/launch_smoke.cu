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

#include <functional>
#include <iostream>
#include <type_traits>

#include "../hierarchy/testing_common.cuh"

struct functor_int_argument
{
  __device__ void operator()(int dummy) {}
};

template <unsigned int BlockSize>
struct functor_taking_config
{
  template <typename Config>
  __device__ void operator()(Config conf, int grid_size)
  {
    static_assert(conf.dims.static_count(cudax::thread, cudax::block) == BlockSize);
    assert(conf.dims.count(cudax::block, cudax::grid) == grid_size);
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
  }
};

__global__ void kernel_int_argument(int dummy) {}

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
  }
};

TEST_CASE("Smoke", "[launch]")
{
  // Use raw stream to make sure it can be implicitly converted on call to launch
  cudaStream_t stream;

  CUDART(cudaStreamCreate(&stream));
  try
  {
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
          cudax::launch(stream, dims_or_conf, kernel_int_argument, 1);
          cudax::launch(stream, dims_or_conf, functor_int_argument(), dummy);
          cudax::launch(stream, dims_or_conf, functor_int_argument(), 1);

          cudax::launch(stream, dims_or_conf, kernel_int_argument, 1U);
          cudax::launch(stream, dims_or_conf, functor_int_argument(), 1U);
        };
        lambda(config);
        lambda(dimensions);
      }
      SECTION("Config argument")
      {
        auto functor_config = functor_taking_config<block_size>();
        auto kernel_config  = kernel_taking_config<decltype(config), block_size>;

        cudax::launch(stream, config, functor_config, grid_size);
        cudax::launch(stream, config, functor_config, ::cuda::std::move(grid_size));

        cudax::launch(stream, config, kernel_config, grid_size);
        cudax::launch(stream, config, kernel_config, ::cuda::std::move(grid_size));

        cudax::launch(stream, config, functor_config, static_cast<unsigned int>(grid_size));
        cudax::launch(stream, config, kernel_config, static_cast<unsigned int>(grid_size));
      }
      SECTION("Dimensions argument")
      {
        auto functor_dims = functor_taking_dims<block_size>();
        auto kernel_dims  = kernel_taking_dims<decltype(dimensions), block_size>;

        cudax::launch(stream, dimensions, functor_dims, grid_size);
        cudax::launch(stream, dimensions, functor_dims, ::cuda::std::move(grid_size));

        cudax::launch(stream, dimensions, kernel_dims, grid_size);
        cudax::launch(stream, dimensions, kernel_dims, ::cuda::std::move(grid_size));

        cudax::launch(stream, dimensions, functor_dims, static_cast<unsigned int>(grid_size));
        cudax::launch(stream, dimensions, kernel_dims, static_cast<unsigned int>(grid_size));
      }
    }
    SECTION("Lambda")
    {
      cudax::launch(stream, cudax::block_dims<256>() & cudax::grid_dims(1), [] __device__(auto dims) {
        if (dims.rank(cudax::thread, cudax::block) == 0)
        {
          printf("Hello from the GPU\n");
        }
      });
    }
    SECTION("Dynamic shared memory option")
    {
      auto dims = cudax::block_dims<32>() & cudax::grid_dims<1>();

      {
        auto conf = cudax::kernel_config(dims, cudax::dynamic_shared_memory<my_dynamic_smem_t>());

        cudax::launch(stream, conf, dynamic_smem_single<my_dynamic_smem_t>());
      }

      {
        const int size = 2;
        auto conf      = cudax::kernel_config(dims, cudax::dynamic_shared_memory<my_dynamic_smem_t>(size));
        cudax::launch(stream, conf, dynamic_smem_span<my_dynamic_smem_t, ::cuda::std::dynamic_extent>(), size);
      }

      {
        constexpr int size = 3;
        auto conf          = cudax::kernel_config(dims, cudax::dynamic_shared_memory<my_dynamic_smem_t, size>());
        cudax::launch(stream, conf, dynamic_smem_span<my_dynamic_smem_t, size>(), size);
      }
    }
    CUDART(cudaStreamSynchronize(stream));
    CUDART(cudaStreamDestroy(stream));
  }
  catch (cuda::cuda_error& e)
  {
    printf("Launch error %s\n", e.what());
    throw std::runtime_error(e.what());
  }
}
