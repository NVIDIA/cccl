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
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

namespace
{
template <cuda::std::size_t... Ns, class Config>
__device__ void test_group_as(Config config)
{
  using NsSeq = cuda::std::integer_sequence<cuda::std::size_t, Ns...>;
  constexpr unsigned ns[]{static_cast<unsigned>(Ns)...};
  constexpr cuda::std::size_t ngroups = sizeof...(Ns);

  cuda::std::size_t group_starts[ngroups];
  cuda::std::exclusive_scan(cuda::std::begin(ns), cuda::std::end(ns), group_starts, cuda::std::size_t{});

  // Test static Ns.
  {
    using Mapping = cudax::group_as<cudax::__group_as_static_tag<Ns...>, true>;

    // Test default constructor.
    {
      static_assert(cuda::std::is_trivially_default_constructible_v<Mapping>);
      static_assert(cuda::std::is_empty_v<Mapping>);

      Mapping mapping;
      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(mapping.count(i) == ns[i]);
      }
    }

    // Test the mapping is constructible from the Ns sequence.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, NsSeq>);

      cudax::group_as mapping{NsSeq{}};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);

      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(mapping.count(i) == ns[i]);
      }
    }

    // Test the mapping is not constructible from Ns sequence and non_exhaustive_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, NsSeq, cudax::non_exhaustive_t>);

    // Test static_group_count().
    static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_group_count())>);
    static_assert(noexcept(Mapping::static_group_count()));
    static_assert(Mapping::static_group_count() == ngroups);

    // Test static_count().
    {
      static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_count(cuda::std::size_t{}))>);
      static_assert(noexcept(Mapping::static_count(cuda::std::size_t{})));
      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(Mapping::static_count(i) == ns[i]);
      }
    }

    // Test is_always_exhaustive().
    static_assert(cuda::std::is_same_v<bool, decltype(Mapping::is_always_exhaustive())>);
    static_assert(noexcept(Mapping::is_always_exhaustive()));
    static_assert(Mapping::is_always_exhaustive());

    // Test count().
    {
      static_assert(
        cuda::std::is_same_v<unsigned, decltype(cuda::std::declval<const Mapping>().count(cuda::std::size_t{}))>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().count(cuda::std::size_t{})));

      const Mapping mapping;
      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(mapping.count(i) == ns[i]);
      }
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

      const auto rank_in_warp = cuda::gpu_thread.rank(parent_group);

      unsigned group_rank_ref = ngroups - 1;
      unsigned rank_ref       = rank_in_warp - group_starts[group_rank_ref];
      for (unsigned i = 1; i < ngroups; ++i)
      {
        if (rank_in_warp < group_starts[i])
        {
          group_rank_ref = i - 1;
          rank_ref       = rank_in_warp - group_starts[i - 1];
          break;
        }
      }

      static_assert(Result::static_group_count() == ngroups);
      CUDAX_CHECK(result.group_count() == static_cast<unsigned>(ngroups));
      CUDAX_CHECK(result.group_rank() == group_rank_ref);

      static_assert(Result::static_count() == cuda::std::dynamic_extent);
      CUDAX_CHECK(result.count() == ns[group_rank_ref]);
      CUDAX_CHECK(result.rank() == rank_ref);

      CUDAX_CHECK(result.is_valid());
      static_assert(Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());
    }
  }

  // Test dynamic Ns.
  {
    using Mapping = cudax::group_as<cudax::__group_as_dynamic_tag<ngroups>, true>;

    // Test default constructor.
    static_assert(!cuda::std::is_default_constructible_v<Mapping>);

    // Test the mapping is constructible from the Ns array.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, decltype(ns)>);

      cudax::group_as mapping{ns};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);

      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(mapping.count(i) == ns[i]);
      }
    }

    // Test the mapping is not constructible from Ns array and non_exhaustive_t.
    static_assert(!cuda::std::is_constructible_v<Mapping, decltype(ns), cudax::non_exhaustive_t>);

    // Test static_group_count().
    static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_group_count())>);
    static_assert(noexcept(Mapping::static_group_count()));
    static_assert(Mapping::static_group_count() == ngroups);

    // Test static_count().
    {
      static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_count(cuda::std::size_t{}))>);
      static_assert(noexcept(Mapping::static_count(cuda::std::size_t{})));
      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(Mapping::static_count(i) == cuda::std::dynamic_extent);
      }
    }

    // Test is_always_exhaustive().
    static_assert(cuda::std::is_same_v<bool, decltype(Mapping::is_always_exhaustive())>);
    static_assert(noexcept(Mapping::is_always_exhaustive()));
    static_assert(Mapping::is_always_exhaustive());

    // Test count().
    {
      static_assert(
        cuda::std::is_same_v<unsigned, decltype(cuda::std::declval<const Mapping>().count(cuda::std::size_t{}))>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().count(cuda::std::size_t{})));

      const Mapping mapping{ns};
      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(mapping.count(i) == ns[i]);
      }
    }

    // Test map(...).
    {
      const cudax::this_warp parent_group{config};

      static_assert(
        cudax::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group))>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group)));

      const Mapping mapping{ns};
      auto result  = mapping.map(cuda::gpu_thread, parent_group);
      using Result = decltype(result);

      const auto rank_in_warp = cuda::gpu_thread.rank_as<unsigned>(parent_group);

      unsigned group_rank_ref = ngroups - 1;
      unsigned rank_ref       = rank_in_warp - group_starts[group_rank_ref];
      for (unsigned i = 1; i < ngroups; ++i)
      {
        if (rank_in_warp < group_starts[i])
        {
          group_rank_ref = i - 1;
          rank_ref       = rank_in_warp - group_starts[i - 1];
          break;
        }
      }

      static_assert(Result::static_group_count() == ngroups);
      CUDAX_CHECK(result.group_count() == static_cast<unsigned>(ngroups));
      CUDAX_CHECK(result.group_rank() == group_rank_ref);

      static_assert(Result::static_count() == cuda::std::dynamic_extent);
      CUDAX_CHECK(result.count() == ns[group_rank_ref]);
      CUDAX_CHECK(result.rank() == rank_ref);

      CUDAX_CHECK(result.is_valid());
      static_assert(Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());
    }
  }
}

template <cuda::std::size_t... Ns, class Config>
__device__ void test_group_as_non_exhaustive(Config config)
{
  using NsSeq = cuda::std::integer_sequence<cuda::std::size_t, Ns...>;
  constexpr unsigned ns[]{static_cast<unsigned>(Ns)...};
  constexpr cuda::std::size_t ngroups = sizeof...(Ns);

  cuda::std::size_t group_starts[ngroups];
  cuda::std::exclusive_scan(cuda::std::begin(ns), cuda::std::end(ns), group_starts, cuda::std::size_t{});

  const auto ns_sum = cuda::std::accumulate(cuda::std::begin(ns), cuda::std::end(ns), 0u);

  // Test static Ns.
  {
    using Mapping = cudax::group_as<cudax::__group_as_static_tag<Ns...>, false>;

    // Test default constructor.
    {
      static_assert(cuda::std::is_trivially_default_constructible_v<Mapping>);
      static_assert(cuda::std::is_empty_v<Mapping>);

      Mapping mapping;
      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(mapping.count(i) == ns[i]);
      }
    }

    // Test the mapping is not constructible from the Ns sequence.
    static_assert(!cuda::std::is_constructible_v<Mapping, NsSeq>);

    // Test the mapping is constructible from Ns sequence and non_exhaustive_t.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, NsSeq, cudax::non_exhaustive_t>);

      cudax::group_as mapping{NsSeq{}, cudax::non_exhaustive};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);

      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(mapping.count(i) == ns[i]);
      }
    }

    // Test static_group_count().
    static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_group_count())>);
    static_assert(noexcept(Mapping::static_group_count()));
    static_assert(Mapping::static_group_count() == ngroups);

    // Test static_count().
    {
      static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_count(cuda::std::size_t{}))>);
      static_assert(noexcept(Mapping::static_count(cuda::std::size_t{})));
      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(Mapping::static_count(i) == ns[i]);
      }
    }

    // Test is_always_exhaustive().
    static_assert(cuda::std::is_same_v<bool, decltype(Mapping::is_always_exhaustive())>);
    static_assert(noexcept(Mapping::is_always_exhaustive()));
    static_assert(!Mapping::is_always_exhaustive());

    // Test count().
    {
      static_assert(
        cuda::std::is_same_v<unsigned, decltype(cuda::std::declval<const Mapping>().count(cuda::std::size_t{}))>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().count(cuda::std::size_t{})));

      const Mapping mapping;
      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(mapping.count(i) == ns[i]);
      }
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

      const auto rank_in_warp = cuda::gpu_thread.rank(parent_group);
      const auto is_valid_ref = (rank_in_warp < ns_sum);

      static_assert(Result::static_group_count() == ngroups);
      static_assert(Result::static_count() == cuda::std::dynamic_extent);
      static_assert(!Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());

      CUDAX_CHECK(result.group_count() == static_cast<unsigned>(ngroups));
      CUDAX_CHECK(result.is_valid() == is_valid_ref);

      if (is_valid_ref)
      {
        unsigned group_rank_ref = ngroups - 1;
        unsigned rank_ref       = rank_in_warp - group_starts[group_rank_ref];
        for (unsigned i = 1; i < ngroups; ++i)
        {
          if (rank_in_warp < group_starts[i])
          {
            group_rank_ref = i - 1;
            rank_ref       = rank_in_warp - group_starts[i - 1];
            break;
          }
        }

        CUDAX_CHECK(result.group_rank() == group_rank_ref);

        CUDAX_CHECK(result.count() == ns[group_rank_ref]);
        CUDAX_CHECK(result.rank() == rank_ref);
      }
    }
  }

  // Test dynamic Ns.
  {
    using Mapping = cudax::group_as<cudax::__group_as_dynamic_tag<ngroups>, false>;

    // Test default constructor.
    static_assert(!cuda::std::is_default_constructible_v<Mapping>);

    // Test the mapping is not constructible from the Ns array.
    static_assert(!cuda::std::is_constructible_v<Mapping, decltype(ns)>);

    // Test the mapping is constructible from Ns array and non_exhaustive_t.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, decltype(ns), cudax::non_exhaustive_t>);

      cudax::group_as mapping{ns, cudax::non_exhaustive};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);

      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(mapping.count(i) == ns[i]);
      }
    }

    // Test static_group_count().
    static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_group_count())>);
    static_assert(noexcept(Mapping::static_group_count()));
    static_assert(Mapping::static_group_count() == ngroups);

    // Test static_count().
    {
      static_assert(cuda::std::is_same_v<cuda::std::size_t, decltype(Mapping::static_count(cuda::std::size_t{}))>);
      static_assert(noexcept(Mapping::static_count(cuda::std::size_t{})));
      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(Mapping::static_count(i) == cuda::std::dynamic_extent);
      }
    }

    // Test is_always_exhaustive().
    static_assert(cuda::std::is_same_v<bool, decltype(Mapping::is_always_exhaustive())>);
    static_assert(noexcept(Mapping::is_always_exhaustive()));
    static_assert(!Mapping::is_always_exhaustive());

    // Test count().
    {
      static_assert(
        cuda::std::is_same_v<unsigned, decltype(cuda::std::declval<const Mapping>().count(cuda::std::size_t{}))>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().count(cuda::std::size_t{})));

      const Mapping mapping{ns, cudax::non_exhaustive};
      for (cuda::std::size_t i = 0; i < ngroups; ++i)
      {
        CUDAX_CHECK(mapping.count(i) == ns[i]);
      }
    }

    // Test map(...).
    {
      const cudax::this_warp parent_group{config};

      static_assert(
        cudax::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group))>);
      static_assert(noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group)));

      const Mapping mapping{ns, cudax::non_exhaustive};
      auto result  = mapping.map(cuda::gpu_thread, parent_group);
      using Result = decltype(result);

      const auto rank_in_warp = cuda::gpu_thread.rank(parent_group);
      const auto is_valid_ref = (rank_in_warp < ns_sum);

      static_assert(Result::static_group_count() == ngroups);
      static_assert(Result::static_count() == cuda::std::dynamic_extent);
      static_assert(!Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());

      CUDAX_CHECK(result.group_count() == static_cast<unsigned>(ngroups));
      CUDAX_CHECK(result.is_valid() == is_valid_ref);

      if (is_valid_ref)
      {
        unsigned group_rank_ref = ngroups - 1;
        unsigned rank_ref       = rank_in_warp - group_starts[group_rank_ref];
        for (unsigned i = 1; i < ngroups; ++i)
        {
          if (rank_in_warp < group_starts[i])
          {
            group_rank_ref = i - 1;
            rank_ref       = rank_in_warp - group_starts[i - 1];
            break;
          }
        }

        CUDAX_CHECK(result.group_rank() == group_rank_ref);

        CUDAX_CHECK(result.count() == ns[group_rank_ref]);
        CUDAX_CHECK(result.rank() == rank_ref);
      }
    }
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_group_as<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
      config);
    test_group_as<2, 4, 8, 16, 2>(config);
    test_group_as<3, 5, 1, 1, 22>(config);
    test_group_as<31, 1>(config);
    test_group_as<32>(config);

    test_group_as_non_exhaustive<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
      config);
    test_group_as_non_exhaustive<2, 4, 8, 16, 2>(config);
    test_group_as_non_exhaustive<3, 5, 1, 1, 22>(config);
    test_group_as_non_exhaustive<31, 1>(config);
    test_group_as_non_exhaustive<32>(config);
    test_group_as_non_exhaustive<31>(config);
    test_group_as_non_exhaustive<4, 6, 8>(config);
    test_group_as_non_exhaustive<2, 2, 3, 1, 14>(config);
  }
};
} // namespace

C2H_TEST("Group-as mapping", "[group]")
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
