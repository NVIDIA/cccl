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
#include <cuda/functional>
#include <cuda/launch>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/detail/probing_scheme_base.cuh>
#include <cuda/experimental/__cuco/detail/utility/hash_to_index.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

namespace cuco = cuda::experimental::cuco;
namespace cg   = cooperative_groups;

using index_types = c2h::type_list<cuda::std::int32_t, cuda::std::uint32_t, cuda::std::int64_t, cuda::std::uint64_t>;

template <class Index>
_CCCL_HOST_DEVICE_API void check_hash_reduction()
{
  using unsigned_index         = cuda::std::make_unsigned_t<Index>;
  constexpr auto i64_min       = cuda::std::numeric_limits<cuda::std::int64_t>::min();
  constexpr auto u64_max       = cuda::std::numeric_limits<cuda::std::uint64_t>::max();
  constexpr auto limit         = cuda::std::numeric_limits<Index>::max();
  constexpr auto wide          = cuda::std::uint64_t{0x100000003};
  constexpr auto min_magnitude = cuda::std::uint64_t{1} << 63;
  constexpr cuda::std::array<cuda::std::uint64_t, 2> array_hash{wide, u64_max};

  static_assert(cuco::detail::__hash_to_index(-1, Index{17}) == Index{1});
  CHECK(cuco::detail::__hash_to_index(-1, Index{17}) == Index{1});
  static_assert(cuco::detail::__hash_to_index(cuda::std::int8_t{-128}, Index{257}) == Index{128});
  CHECK(cuco::detail::__hash_to_index(cuda::std::int8_t{-128}, Index{257}) == Index{128});
  static_assert(cuco::detail::__hash_to_index(cuda::std::uint8_t{255}, Index{257}) == Index{255});
  CHECK(cuco::detail::__hash_to_index(cuda::std::uint8_t{255}, Index{257}) == Index{255});
  static_assert(cuco::detail::__hash_to_index(i64_min, Index{17})
                == static_cast<Index>(static_cast<unsigned_index>(min_magnitude) % 17));
  CHECK(cuco::detail::__hash_to_index(i64_min, Index{17})
        == static_cast<Index>(static_cast<unsigned_index>(min_magnitude) % 17));
  static_assert(cuco::detail::__hash_to_index(i64_min, limit)
                == static_cast<Index>(static_cast<unsigned_index>(min_magnitude) % limit));
  CHECK(cuco::detail::__hash_to_index(i64_min, limit)
        == static_cast<Index>(static_cast<unsigned_index>(min_magnitude) % limit));
  static_assert(
    cuco::detail::__hash_to_index(u64_max, limit) == static_cast<Index>(static_cast<unsigned_index>(u64_max) % limit));
  CHECK(cuco::detail::__hash_to_index(u64_max, limit)
        == static_cast<Index>(static_cast<unsigned_index>(u64_max) % limit));
  static_assert(
    cuco::detail::__hash_to_index(u64_max, Index{17}) == static_cast<Index>(static_cast<unsigned_index>(u64_max) % 17));
  CHECK(cuco::detail::__hash_to_index(u64_max, Index{17})
        == static_cast<Index>(static_cast<unsigned_index>(u64_max) % 17));
  static_assert(
    cuco::detail::__hash_to_index(wide, Index{13}) == static_cast<Index>(static_cast<unsigned_index>(wide) % 13));
  CHECK(cuco::detail::__hash_to_index(wide, Index{13}) == static_cast<Index>(static_cast<unsigned_index>(wide) % 13));
  static_assert(
    cuco::detail::__hash_to_index(array_hash, Index{13}) == static_cast<Index>(static_cast<unsigned_index>(wide) % 13));
  CHECK(cuco::detail::__hash_to_index(array_hash, Index{13})
        == static_cast<Index>(static_cast<unsigned_index>(wide) % 13));
  static_assert(cuco::detail::__hash_to_index(i64_min, Index{1}) == Index{0});
  CHECK(cuco::detail::__hash_to_index(i64_min, Index{1}) == Index{0});

#if _CCCL_HAS_INT128()
  constexpr auto wide128        = (__uint128_t{1} << 100) + 3;
  constexpr auto signed_wide128 = -static_cast<__int128_t>(wide128);
  constexpr auto i128_min       = cuda::std::numeric_limits<__int128_t>::min();
  constexpr auto u128_max       = cuda::std::numeric_limits<__uint128_t>::max();
  static_assert(cuco::detail::__hash_to_index(wide128, Index{13}) == Index{3});
  CHECK(cuco::detail::__hash_to_index(wide128, Index{13}) == Index{3});
  static_assert(cuco::detail::__hash_to_index(signed_wide128, Index{13}) == Index{3});
  CHECK(cuco::detail::__hash_to_index(signed_wide128, Index{13}) == Index{3});
  static_assert(cuco::detail::__hash_to_index(i128_min, Index{17}) == Index{0});
  CHECK(cuco::detail::__hash_to_index(i128_min, Index{17}) == Index{0});
  static_assert(cuco::detail::__hash_to_index(u128_max, Index{17}) == Index{0});
  CHECK(cuco::detail::__hash_to_index(u128_max, Index{17}) == Index{0});
  static_assert(cuco::detail::__hash_to_index(u128_max, limit)
                == static_cast<Index>(static_cast<unsigned_index>(u128_max) % limit));
  CHECK(cuco::detail::__hash_to_index(u128_max, limit)
        == static_cast<Index>(static_cast<unsigned_index>(u128_max) % limit));
  static_assert(cuco::detail::__hash_to_index(i128_min, Index{1}) == Index{0});
  CHECK(cuco::detail::__hash_to_index(i128_min, Index{1}) == Index{0});
#endif // _CCCL_HAS_INT128()
}

template <class Index>
_CCCL_KERNEL_ATTRIBUTES void check_hash_reduction_kernel()
{
  check_hash_reduction<Index>();
}

C2H_TEST("cuco hash reduction handles signed minima and differing widths", "[probing][hash]", index_types)
{
  using index_type = c2h::get<0, TestType>;
  check_hash_reduction<index_type>();
  const cuda::stream stream{cuda::device_ref{0}};
  cuda::launch(
    stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1>()), check_hash_reduction_kernel<index_type>);
  stream.sync();
}

template <class Index, class Capacity>
_CCCL_DEVICE_API void check_wraparound(Capacity capacity)
{
  using dynamic_extent = cuda::std::extents<Index, cuda::std::dynamic_extent>;
  const auto max       = capacity.extent(0);

  // The mathematical sum exceeds the largest index, for both signed and unsigned types.
  cuco::detail::__probing_iterator<Capacity, dynamic_extent> large_step{max - 2, dynamic_extent{max - 3}, capacity};
  const auto previous = large_step++;
  CHECK(*previous == max - 2);
  CHECK(*large_step == max - 5);
  CHECK(*(++large_step) == max - 8);

  cuco::detail::__probing_iterator<Capacity, dynamic_extent> equal_step{max - 1, dynamic_extent{max}, capacity};
  CHECK(*(++equal_step) == max - 1);

  using unit_step = cuda::std::extents<Index, 1>;
  cuco::detail::__probing_iterator<Capacity, unit_step> unit{max - 1, unit_step{}, capacity};
  CHECK(*(++unit) == 0);
  CHECK(*(++unit) == 1);
}

template <class Index>
_CCCL_KERNEL_ATTRIBUTES void check_wraparound_kernel()
{
  constexpr auto max     = cuda::std::numeric_limits<Index>::max();
  using dynamic_capacity = cuda::std::extents<Index, cuda::std::dynamic_extent>;
  // max can be the dynamic_extent sentinel for an unsigned index type.
  using static_capacity = cuda::std::extents<Index, static_cast<cuda::std::size_t>(max - 1)>;
  check_wraparound<Index>(dynamic_capacity{max});
  check_wraparound<Index>(static_capacity{});
}

C2H_TEST("cuco probe advancement wraps without index overflow", "[probing][overflow]", index_types)
{
  using index_type = c2h::get<0, TestType>;
  const cuda::stream stream{cuda::device_ref{0}};
  cuda::launch(
    stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1>()), check_wraparound_kernel<index_type>);
  stream.sync();
}

template <class Hash>
struct constant_hash
{
  Hash value;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr Hash operator()(int) const noexcept
  {
    return value;
  }
};

template <class Iterator>
_CCCL_DEVICE_API void check_sequence(Iterator iterator, unsigned initial, unsigned step, unsigned capacity, int probes)
{
  unsigned visited = 0;
  for (int probe = 0; probe < probes; ++probe)
  {
    const auto expected = (initial + static_cast<unsigned>(probe) * step) % capacity;
    CHECK(*iterator == static_cast<decltype(*iterator)>(expected));
    visited |= 1u << static_cast<unsigned>(*iterator);
    ++iterator;
  }
  // With seven prime groups, every group must be visited exactly once before returning to the start.
  CHECK(*iterator == static_cast<decltype(*iterator)>(initial));
  CHECK(cuda::std::popcount(visited) == probes);
}

template <int CgSize, int BucketSize, class Index, class Hasher>
_CCCL_KERNEL_ATTRIBUTES void
check_probing_kernel(Hasher first_hash, Hasher second_hash, int key, unsigned first, unsigned step)
{
  constexpr unsigned groups = 7;
  constexpr unsigned stride = CgSize * BucketSize;
  const auto tile           = cg::tiled_partition<CgSize>(cg::this_thread_block());
  const unsigned lane       = tile.thread_rank();

  const cuco::linear_probing<CgSize, Hasher> linear{first_hash};
  const cuco::double_hashing<CgSize, Hasher> double_hash{first_hash, second_hash};

  using dynamic_capacity = cuda::std::extents<Index, cuda::std::dynamic_extent>;
  using scalar_capacity  = cuda::std::extents<Index, groups * BucketSize>;
  using group_capacity   = cuda::std::extents<Index, groups * stride>;

  const auto scalar_linear = linear.template make_iterator<BucketSize>(key, dynamic_capacity{groups * BucketSize});
  const auto static_scalar_linear = linear.template make_iterator<BucketSize>(key, scalar_capacity{});
  const auto scalar_double = double_hash.template make_iterator<BucketSize>(key, dynamic_capacity{groups * BucketSize});
  const auto static_scalar_double = double_hash.template make_iterator<BucketSize>(key, scalar_capacity{});

  check_sequence(scalar_linear, first * BucketSize, BucketSize, groups * BucketSize, groups);
  check_sequence(static_scalar_linear, first * BucketSize, BucketSize, groups * BucketSize, groups);
  check_sequence(scalar_double, first * BucketSize, step * BucketSize, groups * BucketSize, groups);
  check_sequence(static_scalar_double, first * BucketSize, step * BucketSize, groups * BucketSize, groups);

  const auto group_linear = linear.template make_iterator<BucketSize>(tile, key, dynamic_capacity{groups * stride});
  const auto static_group_linear = linear.template make_iterator<BucketSize>(tile, key, group_capacity{});
  const auto group_double =
    double_hash.template make_iterator<BucketSize>(tile, key, dynamic_capacity{groups * stride});
  const auto static_group_double = double_hash.template make_iterator<BucketSize>(tile, key, group_capacity{});
  const auto initial             = first * stride + lane * BucketSize;
  check_sequence(group_linear, initial, stride, groups * stride, groups);
  check_sequence(static_group_linear, initial, stride, groups * stride, groups);
  check_sequence(group_double, initial, step * stride, groups * stride, groups);
  check_sequence(static_group_double, initial, step * stride, groups * stride, groups);
}

template <class Index, class Hash, int CgSize, int BucketSize>
void check_probing(cuda::stream_ref stream,
                   Hash first,
                   Hash second,
                   cuda::std::uint64_t first_magnitude,
                   cuda::std::uint64_t second_magnitude)
{
  using unsigned_index    = cuda::std::make_unsigned_t<Index>;
  const auto first_bucket = static_cast<unsigned>(static_cast<unsigned_index>(first_magnitude) % 7);
  const auto step         = static_cast<unsigned>(static_cast<unsigned_index>(second_magnitude) % 6 + 1);
  cuda::launch(
    stream,
    cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<CgSize>()),
    check_probing_kernel<CgSize, BucketSize, Index, constant_hash<Hash>>,
    constant_hash<Hash>{first},
    constant_hash<Hash>{second},
    0,
    first_bucket,
    step);
}

template <class Index, int CgSize, int BucketSize>
void check_hash_types(cuda::stream_ref stream)
{
  using i32              = cuda::std::int32_t;
  using u32              = cuda::std::uint32_t;
  using i64              = cuda::std::int64_t;
  using u64              = cuda::std::uint64_t;
  constexpr auto u64_max = cuda::std::numeric_limits<u64>::max();
  constexpr auto u32_max = cuda::std::numeric_limits<u32>::max();
  check_probing<Index, i32, CgSize, BucketSize>(stream, cuda::std::numeric_limits<i32>::min(), i32{-1}, u64{1} << 31, 1);
  check_probing<Index, u32, CgSize, BucketSize>(stream, u32_max, u32_max - 1, u32_max, u32_max - 1);
  check_probing<Index, i64, CgSize, BucketSize>(stream, cuda::std::numeric_limits<i64>::min(), i64{-1}, u64{1} << 63, 1);
  check_probing<Index, u64, CgSize, BucketSize>(stream, u64_max, u64_max - 1, u64_max, u64_max - 1);
  using array_hash = cuda::std::array<u64, 2>;
  check_probing<Index, array_hash, CgSize, BucketSize>(
    stream, array_hash{u64_max, 1}, array_hash{0x100000003, u64_max}, u64_max, 0x100000003);
}

C2H_TEST("cuco scalar and group probing preserve complete bucket cycles", "[probing][hash][cg]", index_types)
{
  using index_type = c2h::get<0, TestType>;
  const cuda::stream stream{cuda::device_ref{0}};
  check_hash_types<index_type, 1, 1>(stream);
  check_hash_types<index_type, 2, 2>(stream);
  stream.sync();
}

#if _CCCL_HAS_INT128()
C2H_TEST("cuco probing accepts public 128-bit hash results", "[probing][hash][cg]", index_types)
{
  using index_type = c2h::get<0, TestType>;
  using hasher     = cuda::hash<cuda::std::int32_t, cuda::hash_algorithm::murmurhash3_x64_128>;

  // Reference digests for key 42 from the public MurmurHash3_x64_128 hash tests, with seeds 0 and 42.
  constexpr auto first_hash  = (__uint128_t{0xe2d23d6a2bbcb816ull} << 64) | __uint128_t{0x286f48e61c6e34cfull};
  constexpr auto second_hash = (__uint128_t{0xf9e3fe3d853fa768ull} << 64) | __uint128_t{0x1f35a00f446c3666ull};
  constexpr auto first = static_cast<unsigned>(static_cast<cuda::std::make_unsigned_t<index_type>>(first_hash) % 7);
  constexpr auto step = static_cast<unsigned>(static_cast<cuda::std::make_unsigned_t<index_type>>(second_hash) % 6 + 1);

  CHECK(cuco::detail::__hash_to_index(hasher{0}(42), index_type{7}) == static_cast<index_type>(first));
  CHECK(cuco::detail::__hash_to_index(hasher{42}(42), index_type{6}) == static_cast<index_type>(step - 1));
  const cuda::stream stream{cuda::device_ref{0}};
  cuda::launch(
    stream,
    cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1>()),
    check_probing_kernel<1, 1, index_type, hasher>,
    hasher{0},
    hasher{42},
    42,
    first,
    step);
  cuda::launch(
    stream,
    cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<2>()),
    check_probing_kernel<2, 2, index_type, hasher>,
    hasher{0},
    hasher{42},
    42,
    first,
    step);
  stream.sync();
}
#endif // _CCCL_HAS_INT128()

template <class Index>
_CCCL_KERNEL_ATTRIBUTES void check_wide_group_offsets_kernel()
{
  constexpr int bucket_size = 1 << 30;
  constexpr Index stride    = Index{8} * Index{bucket_size};
  const auto tile           = cg::tiled_partition<8>(cg::this_thread_block());
  const auto lane_offset    = static_cast<Index>(tile.thread_rank()) * Index{bucket_size};
  const cuda::std::extents<Index, cuda::std::dynamic_extent> capacity{Index{7} * stride};

  const cuco::linear_probing<8, constant_hash<Index>> linear{constant_hash<Index>{6}};
  const cuco::double_hashing<8, constant_hash<Index>> double_hash{constant_hash<Index>{6}, constant_hash<Index>{5}};
  auto linear_iterator = linear.template make_iterator<bucket_size>(tile, 0, capacity);
  auto double_iterator = double_hash.template make_iterator<bucket_size>(tile, 0, capacity);

  // The stride exceeds int, and lanes 4 through 7 have offsets exceeding uint32_t.
  CHECK(*linear_iterator == Index{6} * stride + lane_offset);
  CHECK(*double_iterator == Index{6} * stride + lane_offset);
  CHECK(*(++linear_iterator) == lane_offset);
  CHECK(*(++double_iterator) == Index{5} * stride + lane_offset);
}

using wide_index_types = c2h::type_list<cuda::std::int64_t, cuda::std::uint64_t>;

C2H_TEST("cuco group probing computes strides and lane offsets at index width",
         "[probing][overflow][cg]",
         wide_index_types)
{
  using index_type = c2h::get<0, TestType>;
  const cuda::stream stream{cuda::device_ref{0}};
  cuda::launch(stream,
               cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<8>()),
               check_wide_group_offsets_kernel<index_type>);
  stream.sync();
}
