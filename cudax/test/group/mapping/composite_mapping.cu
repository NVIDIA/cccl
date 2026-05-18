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
#include <cuda/std/numeric>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

namespace
{
template <class Mapping1, class Mapping2, class Config>
__device__ void test_composite_mapping(const Mapping1& mapping1, const Mapping2& mapping2, Config config)
{
  using Mapping = cudax::composite_mapping<Mapping1, Mapping2>;

  // Test construction from 2 mappings.
  {
    cudax::composite_mapping mapping{mapping1, mapping2};
    static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
    static_assert(cuda::std::is_nothrow_constructible_v<Mapping, Mapping1, Mapping2>
                  == (cuda::std::is_nothrow_copy_constructible_v<Mapping1>
                      && cuda::std::is_nothrow_copy_constructible_v<Mapping2>) );
  }

  // Test get().
  {
    const cudax::composite_mapping mapping{mapping1, mapping2};
    static_assert(cuda::std::is_same_v<decltype(mapping.get()), const cuda::std::tuple<Mapping1, Mapping2>&>);
    static_assert(noexcept(mapping.get()));

    const auto& mapping1_ref = cuda::std::get<0>(mapping.get());
    CUDAX_CHECK(mapping1_ref.count() == 4);

    const auto& mapping2_ref = cuda::std::get<1>(mapping.get());
    CUDAX_CHECK(mapping2_ref.count(0) == 1);
    CUDAX_CHECK(mapping2_ref.count(1) == 3);
  }

  // Test map(...).
  {
    const cudax::this_warp parent_group{config};
    const ThreadsInWarpMappingResult prev_mapping_result;
    const cudax::composite_mapping mapping{mapping1, mapping2};

    static_assert(cudax::__group_mapping_result<decltype(mapping.map(parent_group, prev_mapping_result))>);

    auto result  = mapping.map(parent_group, prev_mapping_result);
    using Result = decltype(result);

    const auto rank_in_warp = cuda::gpu_thread.rank_as<unsigned>(parent_group);

    if constexpr (Mapping1::static_count() != cuda::std::dynamic_extent
                  && Mapping2::static_group_count() != cuda::std::dynamic_extent)
    {
      static_assert(Result::static_group_count() == 16);
    }
    else
    {
      static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
    }
    CUDAX_CHECK(result.group_count() == 16);
    CUDAX_CHECK(result.group_rank() == (rank_in_warp / 4 * 2 + (rank_in_warp % 4 > 0)));

    static_assert(Result::static_count() == cuda::std::dynamic_extent);
    CUDAX_CHECK(result.count() == ((rank_in_warp % 4 > 0) ? 3 : 1));
    CUDAX_CHECK(result.rank() == ((rank_in_warp % 4 > 0) ? (rank_in_warp % 4 - 1) : 0));

    CUDAX_CHECK(result.is_valid());
    static_assert(Result::is_always_exhaustive());
    static_assert(Result::is_always_contiguous());
  }

  // Test operator|.
  {
    auto mapping = mapping1 | mapping2;

    static_assert(cuda::std::is_same_v<Mapping, decltype(mapping)>);
    static_assert(noexcept(mapping1 | mapping2));

    const auto& mapping1_ref = cuda::std::get<0>(mapping.get());
    CUDAX_CHECK(mapping1_ref.count() == 4);

    const auto& mapping2_ref = cuda::std::get<1>(mapping.get());
    CUDAX_CHECK(mapping2_ref.count(0) == 1);
    CUDAX_CHECK(mapping2_ref.count(1) == 3);
  }
  {
    auto mapping = cudax::composite_mapping{mapping1} | mapping2;

    static_assert(cuda::std::is_same_v<Mapping, decltype(mapping)>);
    static_assert(noexcept(cudax::composite_mapping{mapping1} | mapping2));

    const auto& mapping1_ref = cuda::std::get<0>(mapping.get());
    CUDAX_CHECK(mapping1_ref.count() == 4);

    const auto& mapping2_ref = cuda::std::get<1>(mapping.get());
    CUDAX_CHECK(mapping2_ref.count(0) == 1);
    CUDAX_CHECK(mapping2_ref.count(1) == 3);
  }
  {
    auto mapping = mapping1 | cudax::composite_mapping{mapping2};

    static_assert(cuda::std::is_same_v<Mapping, decltype(mapping)>);
    static_assert(noexcept(mapping1 | cudax::composite_mapping{mapping2}));

    const auto& mapping1_ref = cuda::std::get<0>(mapping.get());
    CUDAX_CHECK(mapping1_ref.count() == 4);

    const auto& mapping2_ref = cuda::std::get<1>(mapping.get());
    CUDAX_CHECK(mapping2_ref.count(0) == 1);
    CUDAX_CHECK(mapping2_ref.count(1) == 3);
  }
  {
    auto mapping = cudax::composite_mapping{mapping1} | cudax::composite_mapping{mapping2};

    static_assert(cuda::std::is_same_v<Mapping, decltype(mapping)>);
    static_assert(noexcept(cudax::composite_mapping{mapping1} | cudax::composite_mapping{mapping2}));

    const auto& mapping1_ref = cuda::std::get<0>(mapping.get());
    CUDAX_CHECK(mapping1_ref.count() == 4);

    const auto& mapping2_ref = cuda::std::get<1>(mapping.get());
    CUDAX_CHECK(mapping2_ref.count(0) == 1);
    CUDAX_CHECK(mapping2_ref.count(1) == 3);
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    {
      const cudax::group_by<4> mapping1{};
      const cudax::group_as mapping2{cuda::std::integer_sequence<cuda::std::size_t, 1, 3>{}};
      test_composite_mapping(mapping1, mapping2, config);
    }
    {
      const cudax::group_by mapping1{4};
      const cudax::group_as mapping2{cuda::std::integer_sequence<cuda::std::size_t, 1, 3>{}};
      test_composite_mapping(mapping1, mapping2, config);
    }
    {
      const cudax::group_by<4> mapping1{};
      constexpr unsigned counts2[]{1, 3};
      const cudax::group_as mapping2{counts2};
      test_composite_mapping(mapping1, mapping2, config);
    }
    {
      const cudax::group_by mapping1{4};
      constexpr unsigned counts2[]{1, 3};
      const cudax::group_as mapping2{counts2};
      test_composite_mapping(mapping1, mapping2, config);
    }
  }
};
} // namespace

C2H_TEST("Composite mapping", "[group]")
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
