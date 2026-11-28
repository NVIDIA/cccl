//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__cccl_config>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda/type_traits>
#include <cuda/warp>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

struct WarpsInBlockMappingResult
{
  __device__ static constexpr ::cuda::std::size_t static_group_count()
  {
    return 1;
  }

  __device__ unsigned group_count() const
  {
    return 1;
  }

  __device__ static constexpr cuda::std::size_t static_group_rank()
  {
    return 0;
  }

  __device__ unsigned group_rank() const
  {
    return 0;
  }

  __device__ static constexpr ::cuda::std::size_t static_unit_count()
  {
    return cuda::std::dynamic_extent;
  }

  __device__ static constexpr bool has_uniform_static_unit_count()
  {
    return true;
  }

  __device__ static constexpr cuda::std::size_t static_unit_count(cuda::std::size_t)
  {
    return cuda::std::dynamic_extent;
  }

  __device__ unsigned unit_count() const
  {
    return cuda::warp.count_as<unsigned>(cuda::block);
  }

  __device__ unsigned unit_rank() const
  {
    return cuda::warp.rank_as<unsigned>(cuda::block);
  }

  __device__ cuda::device::lane_mask lane_mask() const
  {
    return cuda::device::lane_mask::all();
  }

  __device__ bool is_valid() const
  {
    return true;
  }

  __device__ static constexpr bool is_always_exhaustive()
  {
    return true;
  }

  __device__ static constexpr bool is_always_contiguous()
  {
    return true;
  }
};

struct MutableBarrierId
{
  __host__ __device__ int operator()(int id)
  {
    return id;
  }
};

template <bool MutableOnly, class Config>
__device__ void test_interwarp_synchronizer(Config config)
{
  const auto& hierarchy = config.hierarchy();

  // Test array input range.
  {
    const int barrier_ids[]{5, 3, 15, 1};

    using Synchronizer = decltype(cudax::coop::interwarp_synchronizer{barrier_ids});

    // Test default constructor.
    static_assert(!cuda::std::is_default_constructible_v<Synchronizer>);

    // Test constructor from range of barrier ids.
    {
      // NOLINTNEXTLINE(readability-redundant-parentheses)
      static_assert(cuda::std::is_constructible_v<Synchronizer, decltype((barrier_ids))>);
      [[maybe_unused]] const cudax::coop::interwarp_synchronizer synchronizer{barrier_ids};
    }

    // Test make_instance(...).
    {
      const cudax::coop::this_block parent_group{config};
      const WarpsInBlockMappingResult prev_mapping_result;

      const cudax::coop::group_by mapping{2};
      using View = cuda::std::conditional_t<MutableOnly, NonConstView<const int>, cuda::std::span<const int, 4>>;
      using RangeSynchronizer     = cudax::coop::interwarp_synchronizer<View>;
      using QualifiedSynchronizer = cuda::std::conditional_t<MutableOnly, RangeSynchronizer, const RangeSynchronizer>;
      QualifiedSynchronizer synchronizer{View{barrier_ids, 4}};

      const auto mapping_result = mapping.map(cuda::warp, parent_group, prev_mapping_result);
      static_assert(
        can_make_synchronizer_instance<const RangeSynchronizer,
                                       cuda::warp_level,
                                       decltype(parent_group),
                                       decltype(mapping_result)>
        == !MutableOnly);
      auto synchronizer_instance = synchronizer.make_instance(cuda::warp, parent_group, mapping_result);

      const auto operands = cuda::std::bit_cast<cuda::std::array<unsigned, 2>>(synchronizer_instance);
      REQUIRE(operands[0] == static_cast<unsigned>(barrier_ids[mapping_result.group_rank()]));
      REQUIRE(operands[1] == mapping_result.unit_count() * 32);

      using SynchronizerInstance = decltype(synchronizer_instance);

      static_assert(cuda::std::is_void_v<typename SynchronizerInstance::__packed_barrier_ids_type>);
      static_assert(cuda::is_trivially_copyable_v<SynchronizerInstance>);
      static_assert(cuda::std::is_trivially_destructible_v<SynchronizerInstance>);

      using ArrivalToken = typename SynchronizerInstance::arrival_token;

      // Test arrival_token type.
      static_assert(cuda::std::is_move_constructible_v<ArrivalToken>);
      static_assert(cuda::std::is_move_assignable_v<ArrivalToken>);
      static_assert(cuda::std::is_destructible_v<ArrivalToken>);

      // Test do_sync(...).
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync(hierarchy))>);
      synchronizer_instance.do_sync(hierarchy);

      // Test do_arrive(...).
      static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive(hierarchy))>);
      auto arrival_token = synchronizer_instance.do_arrive(hierarchy);

      // Test do_wait(...).
      static_assert(
        // NOLINTNEXTLINE(cccl-simplify-type-traits)
        cuda::std::is_same_v<void, decltype(synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token)))>);
      synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token));

      // Test do_sync_aligned(...).
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync_aligned(hierarchy))>);
      synchronizer_instance.do_sync_aligned(hierarchy);

      // Test do_arrive_aligned(...).
      static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive_aligned(hierarchy))>);
      auto arrival_token_a = synchronizer_instance.do_arrive_aligned(hierarchy);

      // Test do_wait_aligned(...).
      static_assert(
        cuda::std:: // NOLINT(cccl-simplify-type-traits)
        is_same_v<void, decltype(synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a)))>);
      synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a));

      // Test deinit(...);
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.deinit(mapping_result, hierarchy))>);
      synchronizer_instance.deinit(mapping_result, hierarchy);
    }
  }

  // Test infinite input range.
  {
    const auto barrier_ids = cuda::std::views::iota(1);

    using Synchronizer = decltype(cudax::coop::interwarp_synchronizer{barrier_ids});

    // Test default constructor.
    static_assert(!cuda::std::is_default_constructible_v<Synchronizer>);

    // Test constructor from range of barrier ids.
    {
      static_assert(cuda::std::is_constructible_v<Synchronizer, decltype(barrier_ids)>);
      [[maybe_unused]] const cudax::coop::interwarp_synchronizer synchronizer{barrier_ids};
    }

    // Test make_instance(...).
    {
      const cudax::coop::this_block parent_group{config};
      const WarpsInBlockMappingResult prev_mapping_result;

      const cudax::coop::group_by mapping{2};
      auto ids = [&] {
        if constexpr (MutableOnly)
        {
          return cuda::std::views::transform(barrier_ids, MutableBarrierId{});
        }
        else
        {
          return barrier_ids;
        }
      }();
      using RangeSynchronizer     = cudax::coop::interwarp_synchronizer<decltype(ids)>;
      using QualifiedSynchronizer = cuda::std::conditional_t<MutableOnly, RangeSynchronizer, const RangeSynchronizer>;
      QualifiedSynchronizer synchronizer{cuda::std::move(ids)};

      const auto mapping_result = mapping.map(cuda::warp, parent_group, prev_mapping_result);
      static_assert(
        can_make_synchronizer_instance<const RangeSynchronizer,
                                       cuda::warp_level,
                                       decltype(parent_group),
                                       decltype(mapping_result)>
        == !MutableOnly);
      auto synchronizer_instance = synchronizer.make_instance(cuda::warp, parent_group, mapping_result);

      // Test that synchronizer instance has the expected barrier id.
      REQUIRE((cuda::std::bit_cast<cuda::std::array<unsigned, 2>>(synchronizer_instance)[0]
               == mapping_result.group_rank() + 1));

      using SynchronizerInstance = decltype(synchronizer_instance);

      static_assert(cuda::is_trivially_copyable_v<SynchronizerInstance>);
      static_assert(cuda::std::is_trivially_destructible_v<SynchronizerInstance>);

      using ArrivalToken = typename SynchronizerInstance::arrival_token;

      // Test arrival_token type.
      static_assert(cuda::std::is_move_constructible_v<ArrivalToken>);
      static_assert(cuda::std::is_move_assignable_v<ArrivalToken>);
      static_assert(cuda::std::is_destructible_v<ArrivalToken>);

      // Test do_sync(...).
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync(hierarchy))>);
      synchronizer_instance.do_sync(hierarchy);

      // Test do_arrive(...).
      static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive(hierarchy))>);
      auto arrival_token = synchronizer_instance.do_arrive(hierarchy);

      // Test do_wait(...).
      static_assert(
        // NOLINTNEXTLINE(cccl-simplify-type-traits)
        cuda::std::is_same_v<void, decltype(synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token)))>);
      synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token));

      // Test do_sync_aligned(...).
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync_aligned(hierarchy))>);
      synchronizer_instance.do_sync_aligned(hierarchy);

      // Test do_arrive_aligned(...).
      static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive_aligned(hierarchy))>);
      auto arrival_token_a = synchronizer_instance.do_arrive_aligned(hierarchy);

      // Test do_wait_aligned(...).
      static_assert(
        cuda::std:: // NOLINT(cccl-simplify-type-traits)
        is_same_v<void, decltype(synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a)))>);
      synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a));

      // Test deinit(...);
      // NOLINTNEXTLINE(cccl-simplify-type-traits)
      static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.deinit(mapping_result, hierarchy))>);
      synchronizer_instance.deinit(mapping_result, hierarchy);
    }
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_interwarp_synchronizer<false>(config);
    test_interwarp_synchronizer<true>(config);
  }
};

template <class Block, class Ids>
__device__ void test_packed_barrier_ids(const Block& block, Ids ids)
{
  const cudax::coop::interwarp_synchronizer synchronizer{ids};
  const cudax::coop::virtual_group warps{cuda::warp, block, cudax::coop::group_by{cuda::std::cw<1>}};
  const auto instance = synchronizer.make_instance(cuda::warp, block, warps.__mapping_result());
  using PackedIds     = cuda::std::integral_constant<cuda::std::uint64_t, 0x89AB'CDEF'0123'4567ULL>;
  static_assert(cuda::std::is_same_v<typename decltype(instance)::__packed_barrier_ids_type, PackedIds>);

  // Rank 7 selects ID zero, and rank 15 uses the high bit of the last nibble.
  const auto operands = cuda::std::bit_cast<cuda::std::array<unsigned, 2>>(instance);
  CHECK(operands[0] == (warps.rank(block) ^ 7u));
  CHECK(operands[1] == 32);

  warps.dispatch(cudax::coop::by_group_rank, [&](auto view) {
    constexpr auto rank = view.static_rank(cudax::coop::parent_group);
    const auto ranked   = synchronizer.make_instance(cuda::warp, block, view.__mapping_result());
    using Expected      = cudax::coop::__interwarp_synchronizer_instance<rank ^ 7u, 32, PackedIds>;
    static_assert(cuda::std::is_same_v<decltype(ranked), const Expected>);
    CHECK(operands[0] == (rank ^ 7u));
    ranked.do_sync(block.hierarchy());
  });
}

struct packed_barrier_ids_kernel
{
  template <class Config>
  __device__ void operator()(const Config& config) const
  {
    const cudax::coop::this_block block{config};
    const cudax::coop::virtual_group all_warps{cuda::warp, block, cudax::coop::identity_mapping{}};
    const cudax::coop::interwarp_synchronizer zero{cuda::std::index_sequence<0>{}};
    const auto single = zero.make_instance(cuda::warp, block, all_warps.__mapping_result());
    using PackedZero  = cuda::std::integral_constant<cuda::std::uint32_t, 0>;
    static_assert(cuda::std::is_same_v<typename decltype(single)::__packed_barrier_ids_type, PackedZero>);
    single.do_sync(block.hierarchy());

    // Excluded warps still construct an instance with a nonzero, statically known barrier ID.
    const auto partial = cudax::coop::make_group<cudax::coop::generic_group>(
      cudax::coop::non_exhaustive,
      cuda::warp,
      block,
      cudax::coop::take{2},
      cudax::coop::interwarp_synchronizer{cuda::std::index_sequence<7>{}});
    CHECK(partial.has_value() == (cuda::warp.rank(block) < 2));
    if (partial)
    {
      const auto operands = cuda::std::bit_cast<cuda::std::array<unsigned, 2>>(partial->__synchronizer_instance());
      CHECK(operands[0] == 7);
      CHECK(operands[1] == 64);
      partial->sync();
    }
    block.sync();

    test_packed_barrier_ids(block,
                            cuda::std::integer_sequence<int, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8>{});
#if _CCCL_STD_VER >= 2020
    test_packed_barrier_ids(
      block, cuda::std::constant_wrapper<cuda::std::array{7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8}>{});
#endif // _CCCL_STD_VER >= 2020
  }
};

C2H_TEST("Barrier synchronizer", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<256>());
    cuda::launch(stream, config, TestKernel{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(dim3{256}));
    cuda::launch(stream, config, TestKernel{});
  }
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<512>()), packed_barrier_ids_kernel{});

  stream.sync();
}
