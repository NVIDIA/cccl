//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

namespace
{
template <cuda::std::size_t N, class Config>
__device__ void test_group_by(Config config)
{
  // Test static N.
  {
    using Mapping = cudax::group_by<N>;
    static_assert(cuda::std::is_same_v<Mapping, cudax::group_by<N, true>>);

    // Test default constructor.
    {
      static_assert(cuda::std::is_trivially_default_constructible_v<Mapping>);
      static_assert(cuda::std::is_empty_v<Mapping>);

      cudax::group_by<N> mapping;
      CUDAX_CHECK(mapping.count() == static_cast<unsigned>(N));
    }

    // Test the mapping is not constructible from unsigned.
    static_assert(!cuda::std::is_constructible_v<Mapping, unsigned>);

    // Test the mapping is not constructible from non_exhaustive_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, cudax::non_exhaustive_t>);

    // Test the mapping is not constructible from unsigned and non_exhaustive_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, unsigned, cudax::non_exhaustive_t>);

    // Test static_count().
    static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_count())>);
    static_assert(noexcept(Mapping::static_count()));
    static_assert(Mapping::static_count() == N);

    // Test is_always_exhaustive().
    static_assert(cuda::std::is_same_v<bool, decltype(Mapping::is_always_exhaustive())>);
    static_assert(noexcept(Mapping::is_always_exhaustive()));
    static_assert(Mapping::is_always_exhaustive());

    // Test count().
    {
      static_assert(cuda::std::is_same_v<unsigned, decltype(cuda::std::declval<const Mapping>().count())>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().count()));

      const Mapping mapping;
      CUDAX_CHECK(mapping.count() == static_cast<unsigned>(N));
    }

    // Test map(...).
    {
      const cudax::this_warp parent_group{config};

      static_assert(
        cudax::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group))>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group)));

      const Mapping mapping;
      auto result  = mapping.map(cuda::gpu_thread, parent_group);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == 32 / N);
      CUDAX_CHECK(result.group_count() == cuda::gpu_thread.count(cuda::warp) / N);
      CUDAX_CHECK(result.group_rank() == cuda::gpu_thread.rank(cuda::warp) / N);

      static_assert(Result::static_count() == N);
      CUDAX_CHECK(result.count() == N);
      CUDAX_CHECK(result.rank() == cuda::gpu_thread.rank(cuda::warp) % N);

      CUDAX_CHECK(result.is_valid());
      static_assert(Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());
    }
  }

  // Test dynamic N.
  {
    using Mapping = cudax::group_by<>;
    static_assert(cuda::std::is_same_v<Mapping, cudax::group_by<cuda::std::dynamic_extent, true>>);

    // Test default constructor.
    static_assert(!cuda::std::is_default_constructible_v<Mapping>);
    static_assert(!cuda::std::is_empty_v<Mapping>);

    // Test the mapping is constructible from unsigned.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, unsigned>);

      cudax::group_by mapping{N};
      static_assert(cuda::std::is_same_v<Mapping, decltype(mapping)>);
      CUDAX_CHECK(mapping.count() == static_cast<unsigned>(N));
    }

    // Test the mapping is not constructible from non_exhaustive_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, cudax::non_exhaustive_t>);

    // Test the mapping is not constructible from unsigned and non_exhaustive_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, unsigned, cudax::non_exhaustive_t>);

    // Test static_count().
    static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_count())>);
    static_assert(noexcept(Mapping::static_count()));
    static_assert(Mapping::static_count() == cuda::std::dynamic_extent);

    // Test is_always_exhaustive().
    static_assert(cuda::std::is_same_v<bool, decltype(Mapping::is_always_exhaustive())>);
    static_assert(noexcept(Mapping::is_always_exhaustive()));
    static_assert(Mapping::is_always_exhaustive());

    // Test count().
    {
      static_assert(cuda::std::is_same_v<unsigned, decltype(cuda::std::declval<const Mapping>().count())>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().count()));

      const Mapping mapping{N};
      CUDAX_CHECK(mapping.count() == static_cast<unsigned>(N));
    }

    // Test map(...).
    {
      const cudax::this_warp parent_group{config};

      static_assert(
        cudax::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group))>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group)));

      const Mapping mapping{N};
      auto result  = mapping.map(cuda::gpu_thread, parent_group);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
      CUDAX_CHECK(result.group_count() == cuda::gpu_thread.count(cuda::warp) / N);
      CUDAX_CHECK(result.group_rank() == cuda::gpu_thread.rank(cuda::warp) / N);

      static_assert(Result::static_count() == cuda::std::dynamic_extent);
      CUDAX_CHECK(result.count() == N);
      CUDAX_CHECK(result.rank() == cuda::gpu_thread.rank(cuda::warp) % N);

      CUDAX_CHECK(result.is_valid());
      static_assert(Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());
    }
  }
}

template <cuda::std::size_t N, class Config>
__device__ void test_group_by_non_exhaustive(Config config)
{
  // Test static N.
  {
    using Mapping = cudax::group_by<N, false>;

    // Test default constructor.
    static_assert(cuda::std::is_trivially_default_constructible_v<Mapping>);
    static_assert(cuda::std::is_empty_v<Mapping>);

    // Test the mapping is not constructible from unsigned.
    static_assert(!cuda::std::is_constructible_v<Mapping, unsigned>);

    // Test the mapping is not constructible from non_exhaustive_t.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, cudax::non_exhaustive_t>);

      Mapping mapping{cudax::non_exhaustive};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
      CUDAX_CHECK(mapping.count() == static_cast<unsigned>(N));
    }

    // Test the mapping is not constructible from unsigned and non_exhaustive_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, unsigned, cudax::non_exhaustive_t>);

    // Test static_count().
    static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_count())>);
    static_assert(noexcept(Mapping::static_count()));
    static_assert(Mapping::static_count() == N);

    // Test is_always_exhaustive().
    static_assert(cuda::std::is_same_v<bool, decltype(Mapping::is_always_exhaustive())>);
    static_assert(noexcept(Mapping::is_always_exhaustive()));
    static_assert(!Mapping::is_always_exhaustive());

    // Test count().
    {
      static_assert(cuda::std::is_same_v<unsigned, decltype(cuda::std::declval<const Mapping>().count())>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().count()));

      const Mapping mapping{cudax::non_exhaustive};
      CUDAX_CHECK(mapping.count() == static_cast<unsigned>(N));
    }

    // Test map(...).
    {
      const cudax::this_warp parent_group{config};

      static_assert(
        cudax::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group))>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group)));

      const Mapping mapping{cudax::non_exhaustive};
      auto result  = mapping.map(cuda::gpu_thread, parent_group);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == 32 / N);
      static_assert(Result::static_count() == N);
      static_assert(!Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());

      const auto is_valid_ref = cuda::gpu_thread.rank(cuda::warp) < (cuda::gpu_thread.count(cuda::warp) / N) * N;
      CUDAX_CHECK(result.is_valid() == is_valid_ref);

      if (is_valid_ref)
      {
        CUDAX_CHECK(result.group_count() == cuda::gpu_thread.count(cuda::warp) / N);
        CUDAX_CHECK(result.group_rank() == cuda::gpu_thread.rank(cuda::warp) / N);

        CUDAX_CHECK(result.count() == N);
        CUDAX_CHECK(result.rank() == cuda::gpu_thread.rank(cuda::warp) % N);
      }
    }
  }

  // Test dynamic N.
  {
    using Mapping = cudax::group_by<cuda::std::dynamic_extent, false>;

    // Test default constructor.
    static_assert(!cuda::std::is_default_constructible_v<Mapping>);
    static_assert(!cuda::std::is_empty_v<Mapping>);

    // Test the mapping is constructible from unsigned.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, unsigned>);

      Mapping mapping{N};
      static_assert(cuda::std::is_same_v<Mapping, decltype(mapping)>);
      CUDAX_CHECK(mapping.count() == static_cast<unsigned>(N));
    }

    // Test the mapping is not constructible from non_exhaustive_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, cudax::non_exhaustive_t>);

    // Test the mapping is not constructible from unsigned and non_exhaustive_t.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, unsigned, cudax::non_exhaustive_t>);

      cudax::group_by mapping{static_cast<unsigned>(N), cudax::non_exhaustive};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
      CUDAX_CHECK(mapping.count() == static_cast<unsigned>(N));
    }

    // Test static_count().
    static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_count())>);
    static_assert(noexcept(Mapping::static_count()));
    static_assert(Mapping::static_count() == cuda::std::dynamic_extent);

    // Test count().
    {
      static_assert(cuda::std::is_same_v<unsigned, decltype(cuda::std::declval<const Mapping>().count())>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().count()));

      const Mapping mapping{N, cudax::non_exhaustive};
      CUDAX_CHECK(mapping.count() == static_cast<unsigned>(N));
    }

    // Test map(...).
    {
      const cudax::this_warp parent_group{config};

      static_assert(
        cudax::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group))>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group)));

      const Mapping mapping{N, cudax::non_exhaustive};
      auto result  = mapping.map(cuda::gpu_thread, parent_group);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
      static_assert(Result::static_count() == cuda::std::dynamic_extent);
      static_assert(!Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());

      const auto is_valid_ref = cuda::gpu_thread.rank(cuda::warp) < (cuda::gpu_thread.count(cuda::warp) / N) * N;
      CUDAX_CHECK(result.is_valid() == is_valid_ref);

      if (is_valid_ref)
      {
        CUDAX_CHECK(result.group_count() == cuda::gpu_thread.count(cuda::warp) / N);
        CUDAX_CHECK(result.group_rank() == cuda::gpu_thread.rank(cuda::warp) / N);

        CUDAX_CHECK(result.count() == N);
        CUDAX_CHECK(result.rank() == cuda::gpu_thread.rank(cuda::warp) % N);
      }
    }
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_group_by<1>(config);
    test_group_by<2>(config);
    test_group_by<4>(config);
    test_group_by<16>(config);
    test_group_by<32>(config);

    test_group_by_non_exhaustive<1>(config);
    test_group_by_non_exhaustive<2>(config);
    test_group_by_non_exhaustive<3>(config);
    test_group_by_non_exhaustive<4>(config);
    test_group_by_non_exhaustive<14>(config);
    test_group_by_non_exhaustive<16>(config);
    test_group_by_non_exhaustive<30>(config);
    test_group_by_non_exhaustive<32>(config);
  }
};
} // namespace

C2H_TEST("Group-by mapping", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<8, 4>());
    cuda::launch(stream, config, TestKernel{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(dim3{8, 4}));
    cuda::launch(stream, config, TestKernel{});
  }

  stream.sync();
}
