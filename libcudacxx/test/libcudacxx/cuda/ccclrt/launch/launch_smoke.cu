//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/atomic>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/memory>
#include <cuda/stream>

#include <cooperative_groups.h>
#include <testing.cuh>

#include "test_macros.h"

#if !_CCCL_CUDA_COMPILER(CLANG)

__managed__ bool kernel_run_proof = false;

void check_kernel_run(cudaStream_t stream)
{
  CUDART(cudaStreamSynchronize(stream));
  CCCLRT_CHECK(kernel_run_proof);
  kernel_run_proof = false;
}

struct kernel_run_proof_check
{
  TEST_DEVICE_FUNC void operator()()
  {
    CCCLRT_CHECK_DEVICE(kernel_run_proof);
    kernel_run_proof = false;
  }
};

struct functor_int_argument
{
  TEST_DEVICE_FUNC void operator()(int dummy)
  {
    kernel_run_proof = true;
  }
};

template <unsigned int BlockSize>
struct functor_taking_config
{
  template <typename Config>
  TEST_DEVICE_FUNC void operator()(Config config, int grid_size)
  {
    static_assert(cuda::gpu_thread.count(cuda::block, config) == BlockSize);
    CCCLRT_REQUIRE_DEVICE(cuda::block.count(cuda::grid, config) == grid_size);
    kernel_run_proof = true;
  }
};

__global__ void kernel_no_arguments()
{
  kernel_run_proof = true;
}

__global__ void kernel_int_argument(int dummy)
{
  kernel_run_proof = true;
}

template <typename Config, unsigned int BlockSize>
__global__ void kernel_taking_config(Config config, int grid_size)
{
  functor_taking_config<BlockSize>()(config, grid_size);
}

struct my_dynamic_smem_t
{
  int i;
};

template <typename SmemType>
struct dynamic_smem_single
{
  template <typename Config>
  TEST_DEVICE_FUNC void operator()(Config config)
  {
    decltype(auto) dynamic_smem = cuda::dynamic_shared_memory(config);
    static_assert(::cuda::std::is_same_v<SmemType&, decltype(dynamic_smem)>);
    CCCLRT_REQUIRE_DEVICE(::cuda::device::is_object_from(dynamic_smem, ::cuda::device::address_space::shared));
    kernel_run_proof = true;
  }
};

template <typename SmemType, size_t Extent>
struct dynamic_smem_span
{
  template <typename Config>
  TEST_DEVICE_FUNC void operator()(Config config, int size)
  {
    auto dynamic_smem = cuda::dynamic_shared_memory(config);
    static_assert(decltype(dynamic_smem)::extent == Extent);
    static_assert(::cuda::std::is_same_v<SmemType&, decltype(dynamic_smem[1])>);
    CCCLRT_REQUIRE_DEVICE(dynamic_smem.size() == size);
    CCCLRT_REQUIRE_DEVICE(::cuda::device::is_object_from(dynamic_smem[1], ::cuda::device::address_space::shared));
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
      // Disabled for now because we don't handle it with graphs
      // CUDAX_CHECK_FALSE(kernel_run_proof);
    }

    // Immovable to ensure that launch_transform doesn't copy the returned
    // object
    int_convertible(int_convertible&&) noexcept = delete;

    ~int_convertible() noexcept
    {
      // Check that the destructor runs after the kernel is launched
      // Disabled for now because we don't handle it with graphs
      // CUDART(cudaStreamSynchronize(stream_));
      // CCCLRT_CHECK(kernel_run_proof);
    }

    // This is the value that will be passed to the kernel
    int transformed_argument() const
    {
      return value_;
    }
  };

  [[nodiscard]] friend int_convertible
  transform_launch_argument(::cuda::stream_ref stream, launch_transform_to_int_convertible self) noexcept
  {
    return int_convertible(stream.get(), self.value_);
  }
};

// Needs a separate function for Windows extended lambda
void launch_smoke_test(cudaStream_t dst)
{
  cuda::__ensure_current_context guard(cuda::device_ref{0});
  // Use raw stream to make sure it can be implicitly converted on call to
  // launch
  cudaStream_t stream;

  CUDART(cudaStreamCreate(&stream));
  // Spell out all overloads to make sure they compile, include a check for
  // implicit conversions
  {
    const int grid_size      = 4;
    constexpr int block_size = 256;
    auto dimensions          = cuda::make_hierarchy(cuda::grid_dims(grid_size), cuda::block_dims<256>());
    auto config              = cuda::make_config(dimensions);

    // Not taking dims
    {
      cuda::launch(dst, config, kernel_no_arguments);
      check_kernel_run(dst);

      const int dummy = 1;
      cuda::launch(dst, config, kernel_int_argument, dummy);
      check_kernel_run(dst);
      cuda::launch(dst, config, kernel_int_argument, 1);
      check_kernel_run(dst);
      cuda::launch(dst, config, kernel_int_argument, launch_transform_to_int_convertible{1});
      check_kernel_run(dst);
      cuda::launch(dst, config, kernel_int_argument, 1U);
      check_kernel_run(dst);

      cuda::launch(dst, config, functor_int_argument(), dummy);
      check_kernel_run(dst);
      cuda::launch(dst, config, functor_int_argument(), 1);
      check_kernel_run(dst);
      cuda::launch(dst, config, functor_int_argument(), launch_transform_to_int_convertible{1});
      check_kernel_run(dst);
      cuda::launch(dst, config, functor_int_argument(), 1U);
      check_kernel_run(dst);
    }

    // Config argument
    {
      auto functor_instance = functor_taking_config<block_size>();
      auto kernel_instance  = kernel_taking_config<decltype(config), block_size>;

      cuda::launch(dst, config, functor_instance, grid_size);
      check_kernel_run(dst);
      cuda::launch(dst, config, functor_instance, ::cuda::std::move(grid_size));
      check_kernel_run(dst);
      cuda::launch(dst, config, functor_instance, launch_transform_to_int_convertible{grid_size});
      check_kernel_run(dst);
      cuda::launch(dst, config, functor_instance, static_cast<unsigned int>(grid_size));
      check_kernel_run(dst);

      cuda::launch(dst, config, kernel_instance, grid_size);
      check_kernel_run(dst);
      cuda::launch(dst, config, kernel_instance, ::cuda::std::move(grid_size));
      check_kernel_run(dst);
      cuda::launch(dst, config, kernel_instance, launch_transform_to_int_convertible{grid_size});
      check_kernel_run(dst);
      cuda::launch(dst, config, kernel_instance, static_cast<unsigned int>(grid_size));
      check_kernel_run(dst);
    }
  }

  // Dynamic shared memory option
  {
    auto config = cuda::block_dims<32>() & cuda::grid_dims<1>();

    auto test = [&](const auto& input_config) {
      // Single element
      {
        auto config = input_config.add(cuda::dynamic_shared_memory<my_dynamic_smem_t>());

        cuda::launch(dst, config, dynamic_smem_single<my_dynamic_smem_t>());
        check_kernel_run(dst);
      }

      // Dynamic span
      {
        const int size = 2;
        auto config    = input_config.add(cuda::dynamic_shared_memory<my_dynamic_smem_t[]>(size));
        cuda::launch(dst, config, dynamic_smem_span<my_dynamic_smem_t, ::cuda::std::dynamic_extent>(), size);
        check_kernel_run(dst);
      }

      // Static span
      {
        constexpr int size = 3;
        auto config        = input_config.add(cuda::dynamic_shared_memory<my_dynamic_smem_t[size]>());
        cuda::launch(dst, config, dynamic_smem_span<my_dynamic_smem_t, size>(), size);
        check_kernel_run(dst);
      }
    };

    test(config);
    test(config.add(cuda::cooperative_launch(), cuda::launch_priority(0)));
  }
}

C2H_CCCLRT_TEST("Launch smoke stream", "[launch]")
{
  // Use raw stream to make sure it can be implicitly converted on call to
  // launch
  cudaStream_t stream;

  {
    ::cuda::__ensure_current_context guard(cuda::device_ref{0});
    CUDART(cudaStreamCreate(&stream));
  }

  launch_smoke_test(stream);

  {
    ::cuda::__ensure_current_context guard(cuda::device_ref{0});
    CUDART(cudaStreamSynchronize(stream));
    CUDART(cudaStreamDestroy(stream));
  }
}

template <typename DefaultConfig>
struct kernel_with_default_config
{
  DefaultConfig config;

  kernel_with_default_config(DefaultConfig c)
      : config(c)
  {}

  DefaultConfig default_config() const
  {
    return config;
  }

  template <typename Config, typename ConfigCheckFn>
  TEST_DEVICE_FUNC void operator()(Config config, ConfigCheckFn check_fn)
  {
    check_fn(config);
  }
};

struct verify_callable
{
  template <typename Config>
  TEST_DEVICE_FUNC void operator()(Config config)
  {
    static_assert(cuda::gpu_thread.count(cuda::block, config) == 256);
    CCCLRT_REQUIRE(cuda::block.count(cuda::grid, config) == 4);
    cooperative_groups::this_grid().sync();
  }
};

C2H_CCCLRT_TEST("Launch with default config", "")
{
  cuda::stream stream{cuda::device_ref{0}};
  auto grid  = cuda::grid_dims(4);
  auto block = cuda::block_dims<256>;

  SECTION("Combine with empty")
  {
    kernel_with_default_config kernel{cuda::make_config(block, grid, cuda::cooperative_launch())};
    static_assert(cuda::__is_kernel_config<decltype(kernel.default_config())>);
    static_assert(cuda::__kernel_has_default_config<decltype(kernel)>);

    cuda::launch(stream, cuda::make_config(), kernel, verify_callable{});
    stream.sync();
  }
  SECTION("Combine with no overlap")
  {
    kernel_with_default_config kernel{cuda::make_config(block)};
    cuda::launch(stream, cuda::make_config(grid, cuda::cooperative_launch()), kernel, verify_callable{});
    stream.sync();
  }
  SECTION("Combine with overlap")
  {
    kernel_with_default_config kernel{cuda::make_config(cuda::block_dims<1>(), cuda::cooperative_launch())};
    cuda::launch(stream, cuda::make_config(block, grid, cuda::cooperative_launch()), kernel, verify_callable{});
    stream.sync();
  }
}

// Regression test: cuda::launch must work when the calling function has
// a __restrict__-qualified pointer parameter. On some nvcc + host compiler
// combos, __restrict__ survives through the type transformation pipeline
// and causes a function pointer conversion failure in __get_kernel_launcher.
struct restrict_assign_functor
{
  template <typename Config>
  TEST_DEVICE_FUNC void operator()(Config config, int* __restrict__ dst)
  {
    *dst = 42;
  }
};

// The __restrict__ on the function parameter is the trigger: on affected compilers,
// it leaks into the template args of __get_kernel_launcher via cuda::launch.
void launch_with_restrict_param(cuda::stream_ref stream, int* __restrict__ dst)
{
  auto config = cuda::make_config(cuda::grid_dims(1), cuda::block_dims<1>());
  cuda::launch(stream, config, restrict_assign_functor{}, dst);
}

C2H_CCCLRT_TEST("Launch functor with __restrict__ pointer arg", "[launch]")
{
  cuda::stream stream{cuda::device_ref{0}};
  test::pinned<int> val{0};

  launch_with_restrict_param(stream, val.get());
  stream.sync();

  CCCLRT_CHECK(*val == 42);
}

__managed__ cuda::std::size_t launched_nthreads;
__managed__ cuda::std::size_t launched_nblocks;
__managed__ cuda::std::size_t launched_nclusters;

struct LaunchDimsFunctor
{
  template <class Config>
  TEST_DEVICE_FUNC void operator()(const Config& config)
  {
    cuda::atomic_ref(launched_nthreads)++;

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
      cuda::atomic_ref(launched_nblocks)++;

      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                        ({
                          if (__clusterRelativeBlockRank() == 0)
                          {
                            cuda::atomic_ref(launched_nclusters)++;
                          }
                        }),
                        ({ cuda::atomic_ref(launched_nclusters)++; }));
    }
  }
};

template <class GridDesc, class BlockDesc>
void test_launch_dims(cuda::stream_ref stream, GridDesc grid_desc, BlockDesc block_desc)
{
  launched_nthreads  = 0;
  launched_nblocks   = 0;
  launched_nclusters = 0;

  const auto& grid_exts  = grid_desc.extents();
  const auto& block_exts = block_desc.extents();

  const auto config = cuda::make_config(grid_desc, block_desc);
  cuda::launch(stream, config, LaunchDimsFunctor{});
  stream.sync();

  const auto exp_nclusters = cuda::std::size_t{grid_exts.extent(0)} * grid_exts.extent(1) * grid_exts.extent(2);
  CCCLRT_CHECK(launched_nclusters == exp_nclusters);

  const auto exp_nblocks = exp_nclusters;
  CCCLRT_CHECK(launched_nblocks == exp_nblocks);

  const auto exp_nthreads = exp_nblocks * block_exts.extent(0) * block_exts.extent(1) * block_exts.extent(2);
  CCCLRT_CHECK(launched_nthreads == exp_nthreads);
}

template <class GridDesc, class ClusterDesc, class BlockDesc>
void test_launch_dims(cuda::stream_ref stream, GridDesc grid_desc, ClusterDesc cluster_desc, BlockDesc block_desc)
{
  launched_nthreads  = 0;
  launched_nblocks   = 0;
  launched_nclusters = 0;

  const auto& grid_exts    = grid_desc.extents();
  const auto& cluster_exts = cluster_desc.extents();
  const auto& block_exts   = block_desc.extents();

  const auto config = cuda::make_config(grid_desc, cluster_desc, block_desc);
  cuda::launch(stream, config, LaunchDimsFunctor{});
  stream.sync();

  const auto exp_nclusters = cuda::std::size_t{grid_exts.extent(0)} * grid_exts.extent(1) * grid_exts.extent(2);
  CCCLRT_CHECK(launched_nclusters == exp_nclusters);

  const auto exp_nblocks = exp_nclusters * cluster_exts.extent(0) * cluster_exts.extent(1) * cluster_exts.extent(2);
  CCCLRT_CHECK(launched_nblocks == exp_nblocks);

  const auto exp_nthreads = exp_nblocks * block_exts.extent(0) * block_exts.extent(1) * block_exts.extent(2);
  CCCLRT_CHECK(launched_nthreads == exp_nthreads);
}

C2H_TEST("Launch dims", "[launch]")
{
  cuda::stream stream{cuda::device_ref{0}};

  test_launch_dims(stream, cuda::grid_dims(dim3{2}), cuda::block_dims(dim3{10}));
  test_launch_dims(stream, cuda::grid_dims(dim3{2, 9}), cuda::block_dims(dim3{3, 7}));
  test_launch_dims(stream, cuda::grid_dims(dim3{3, 4, 5}), cuda::block_dims(dim3{2, 7, 9}));

  test_launch_dims(stream, cuda::grid_dims(dim3{2}), cuda::block_dims<10>());
  test_launch_dims(stream, cuda::grid_dims(dim3{2, 9}), cuda::block_dims<3, 7>());
  test_launch_dims(stream, cuda::grid_dims(dim3{3, 4, 5}), cuda::block_dims<2, 7, 9>());

  test_launch_dims(stream, cuda::grid_dims<2>(), cuda::block_dims(dim3{10}));
  test_launch_dims(stream, cuda::grid_dims<2, 9>(), cuda::block_dims(dim3{3, 7}));
  test_launch_dims(stream, cuda::grid_dims<3, 4, 5>(), cuda::block_dims(dim3{2, 7, 9}));

  test_launch_dims(stream, cuda::grid_dims<2>(), cuda::block_dims<10>());
  test_launch_dims(stream, cuda::grid_dims<2, 9>(), cuda::block_dims<3, 7>());
  test_launch_dims(stream, cuda::grid_dims<3, 4, 5>(), cuda::block_dims<2, 7, 9>());

  if (cuda::device_attributes::compute_capability_major(stream.device()) >= 9)
  {
    test_launch_dims(stream, cuda::grid_dims(dim3{2}), cuda::cluster_dims(dim3{3}), cuda::block_dims(dim3{10}));
    test_launch_dims(stream, cuda::grid_dims(dim3{2, 9}), cuda::cluster_dims(dim3{1, 5}), cuda::block_dims(dim3{3, 7}));
    test_launch_dims(
      stream, cuda::grid_dims(dim3{3, 4, 5}), cuda::cluster_dims(dim3{3, 1, 2}), cuda::block_dims(dim3{2, 7, 9}));

    test_launch_dims(stream, cuda::grid_dims(dim3{2}), cuda::cluster_dims(dim3{3}), cuda::block_dims<10>());
    test_launch_dims(stream, cuda::grid_dims(dim3{2, 9}), cuda::cluster_dims(dim3{1, 5}), cuda::block_dims<3, 7>());
    test_launch_dims(
      stream, cuda::grid_dims(dim3{3, 4, 5}), cuda::cluster_dims(dim3{3, 1, 2}), cuda::block_dims<2, 7, 9>());

    test_launch_dims(stream, cuda::grid_dims(dim3{2}), cuda::cluster_dims<3>(), cuda::block_dims(dim3{10}));
    test_launch_dims(stream, cuda::grid_dims(dim3{2, 9}), cuda::cluster_dims<1, 5>(), cuda::block_dims(dim3{3, 7}));
    test_launch_dims(
      stream, cuda::grid_dims(dim3{3, 4, 5}), cuda::cluster_dims<3, 1, 2>(), cuda::block_dims(dim3{2, 7, 9}));

    test_launch_dims(stream, cuda::grid_dims(dim3{2}), cuda::cluster_dims<3>(), cuda::block_dims<10>());
    test_launch_dims(stream, cuda::grid_dims(dim3{2, 9}), cuda::cluster_dims<1, 5>(), cuda::block_dims<3, 7>());
    test_launch_dims(stream, cuda::grid_dims(dim3{3, 4, 5}), cuda::cluster_dims<3, 1, 2>(), cuda::block_dims<2, 7, 9>());

    test_launch_dims(stream, cuda::grid_dims<2>(), cuda::cluster_dims(dim3{3}), cuda::block_dims(dim3{10}));
    test_launch_dims(stream, cuda::grid_dims<2, 9>(), cuda::cluster_dims(dim3{1, 5}), cuda::block_dims(dim3{3, 7}));
    test_launch_dims(
      stream, cuda::grid_dims<3, 4, 5>(), cuda::cluster_dims(dim3{3, 1, 2}), cuda::block_dims(dim3{2, 7, 9}));

    test_launch_dims(stream, cuda::grid_dims<2>(), cuda::cluster_dims(dim3{3}), cuda::block_dims<10>());
    test_launch_dims(stream, cuda::grid_dims<2, 9>(), cuda::cluster_dims(dim3{1, 5}), cuda::block_dims<3, 7>());
    test_launch_dims(stream, cuda::grid_dims<3, 4, 5>(), cuda::cluster_dims(dim3{3, 1, 2}), cuda::block_dims<2, 7, 9>());

    test_launch_dims(stream, cuda::grid_dims<2>(), cuda::cluster_dims<3>(), cuda::block_dims(dim3{10}));
    test_launch_dims(stream, cuda::grid_dims<2, 9>(), cuda::cluster_dims<1, 5>(), cuda::block_dims(dim3{3, 7}));
    test_launch_dims(stream, cuda::grid_dims<3, 4, 5>(), cuda::cluster_dims<3, 1, 2>(), cuda::block_dims(dim3{2, 7, 9}));

    test_launch_dims(stream, cuda::grid_dims<2>(), cuda::cluster_dims<3>(), cuda::block_dims<10>());
    test_launch_dims(stream, cuda::grid_dims<2, 9>(), cuda::cluster_dims<1, 5>(), cuda::block_dims<3, 7>());
    test_launch_dims(stream, cuda::grid_dims<3, 4, 5>(), cuda::cluster_dims<3, 1, 2>(), cuda::block_dims<2, 7, 9>());
  }
}

#endif // !_CCCL_CUDA_COMPILER(CLANG)
