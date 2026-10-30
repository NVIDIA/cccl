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
#include <cuda/stream>

#include <cuda/experimental/__cuco/detail/probing_scheme_base.cuh>
#include <cuda/experimental/__cuco/detail/utility/hash_to_index.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

namespace cuco = cuda::experimental::cuco;
namespace cg   = cooperative_groups;

using index_type = cuda::std::size_t;

//! Returns a hash that maps to `index` out of `size` indices, away from the rounding boundaries.
template <class Hash>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr Hash hash_for(index_type index, index_type size)
{
  constexpr auto max = cuda::std::numeric_limits<Hash>::max();
  return static_cast<Hash>(max / size * index + max / size / 2);
}

_CCCL_HOST_DEVICE_API void check_hash_reduction()
{
  using cuco::detail::__hash_to_index;
  constexpr auto u32_max = cuda::std::numeric_limits<cuda::std::uint32_t>::max();
  constexpr auto u64_max = cuda::std::numeric_limits<cuda::std::uint64_t>::max();
  constexpr auto limit   = cuda::std::numeric_limits<index_type>::max();
  constexpr cuda::std::array<cuda::std::uint64_t, 2> array_hash{0, cuda::std::uint64_t{1} << 63};

  // The hash is a fraction of its type's range, so the extremes map to the first and last index.
  static_assert(__hash_to_index(cuda::std::uint64_t{0}, 17) == 0);
  CHECK(__hash_to_index(cuda::std::uint64_t{0}, 17) == 0);
  static_assert(__hash_to_index(u64_max, 17) == 16);
  CHECK(__hash_to_index(u64_max, 17) == 16);
  static_assert(__hash_to_index(u64_max, limit) == limit - 1);
  CHECK(__hash_to_index(u64_max, limit) == limit - 1);
  static_assert(__hash_to_index(u64_max, 1) == 0);
  CHECK(__hash_to_index(u64_max, 1) == 0);
  // Narrower hashes span the same fraction range as 64-bit ones.
  static_assert(__hash_to_index(u32_max, 17) == 16);
  CHECK(__hash_to_index(u32_max, 17) == 16);
  static_assert(__hash_to_index(cuda::std::uint8_t{255}, 257) == 255);
  CHECK(__hash_to_index(cuda::std::uint8_t{255}, 257) == 255);
  static_assert(__hash_to_index(cuda::std::uint32_t{1} << 31, 17) == 8);
  CHECK(__hash_to_index(cuda::std::uint32_t{1} << 31, 17) == 8);
  // The array overload reduces its high word.
  static_assert(__hash_to_index(array_hash, 17) == 8);
  CHECK(__hash_to_index(array_hash, 17) == 8);
}

_CCCL_KERNEL_ATTRIBUTES void check_hash_reduction_kernel()
{
  check_hash_reduction();
}

C2H_TEST("cuco hash reduction maps hashes of any width to valid indices", "[probing][hash]")
{
  check_hash_reduction();
  const cuda::stream stream{cuda::device_ref{0}};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1>()), check_hash_reduction_kernel);
  stream.sync();
}

template <class Capacity>
_CCCL_DEVICE_API void check_wraparound(Capacity capacity)
{
  using dynamic_step = cuda::std::extents<index_type, cuda::std::dynamic_extent>;
  const auto max     = capacity.extent(0);

  // The mathematical sum exceeds the largest index.
  cuco::detail::__probing_iterator<Capacity, dynamic_step> large_step{max - 2, dynamic_step{max - 3}, capacity};
  const auto previous = large_step++;
  CHECK(*previous == max - 2);
  CHECK(*large_step == max - 5);
  CHECK(*(++large_step) == max - 8);

  cuco::detail::__probing_iterator<Capacity, dynamic_step> equal_step{max - 1, dynamic_step{max}, capacity};
  CHECK(*(++equal_step) == max - 1);

  using unit_step = cuda::std::extents<index_type, 1>;
  cuco::detail::__probing_iterator<Capacity, unit_step> unit{max - 1, unit_step{}, capacity};
  CHECK(*(++unit) == 0);
  CHECK(*(++unit) == 1);
}

_CCCL_KERNEL_ATTRIBUTES void check_wraparound_kernel()
{
  constexpr auto max = cuda::std::numeric_limits<index_type>::max();
  // max is the dynamic_extent sentinel, so the largest static capacity is max - 1.
  check_wraparound(cuda::std::extents<index_type, cuda::std::dynamic_extent>{max});
  check_wraparound(cuda::std::extents<index_type, max - 1>{});
}

C2H_TEST("cuco probe advancement wraps without index overflow", "[probing][overflow]")
{
  const cuda::stream stream{cuda::device_ref{0}};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1>()), check_wraparound_kernel);
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
    CHECK(*iterator == expected);
    visited |= 1u << static_cast<unsigned>(*iterator);
    ++iterator;
  }
  // With seven prime groups, every group must be visited exactly once before returning to the start.
  CHECK(*iterator == initial);
  CHECK(cuda::std::popcount(visited) == probes);
}

template <int CgSize, int BucketSize, class Hasher>
_CCCL_KERNEL_ATTRIBUTES void
check_probing_kernel(Hasher first_hash, Hasher second_hash, int key, unsigned first, unsigned step)
{
  constexpr unsigned groups = 7;
  constexpr unsigned stride = CgSize * BucketSize;
  const auto tile           = cg::tiled_partition<CgSize>(cg::this_thread_block());
  const unsigned lane       = tile.thread_rank();

  const cuco::linear_probing<CgSize, Hasher> linear{first_hash};
  const cuco::double_hashing<CgSize, Hasher> double_hash{first_hash, second_hash};

  using dynamic_capacity = cuda::std::extents<index_type, cuda::std::dynamic_extent>;
  using scalar_capacity  = cuda::std::extents<index_type, groups * BucketSize>;
  using group_capacity   = cuda::std::extents<index_type, groups * stride>;

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

template <int CgSize, int BucketSize, class Hasher>
void check_probing(cuda::stream_ref stream, Hasher first, Hasher second, int key, unsigned first_bucket, unsigned step)
{
  cuda::launch(
    stream,
    cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<CgSize>()),
    check_probing_kernel<CgSize, BucketSize, Hasher>,
    first,
    second,
    key,
    first_bucket,
    step);
}

template <class Hash>
void check_hash_type(cuda::stream_ref stream, Hash first, Hash second, unsigned first_bucket, unsigned step)
{
  check_probing<1, 1>(stream, constant_hash<Hash>{first}, constant_hash<Hash>{second}, 0, first_bucket, step);
  check_probing<2, 2>(stream, constant_hash<Hash>{first}, constant_hash<Hash>{second}, 0, first_bucket, step);
}

C2H_TEST("cuco scalar and group probing preserve complete bucket cycles", "[probing][hash][cg]")
{
  using u32        = cuda::std::uint32_t;
  using u64        = cuda::std::uint64_t;
  using array_hash = cuda::std::array<u64, 2>;
  const cuda::stream stream{cuda::device_ref{0}};
  // The first bucket is drawn from seven buckets and the step minus one from six.
  for (const auto [bucket, step] : {cuda::std::pair{0u, 1u}, cuda::std::pair{6u, 6u}})
  {
    check_hash_type(stream, hash_for<u32>(bucket, 7), hash_for<u32>(step - 1, 6), bucket, step);
    check_hash_type(stream, hash_for<u64>(bucket, 7), hash_for<u64>(step - 1, 6), bucket, step);
    check_hash_type(
      stream, array_hash{1, hash_for<u64>(bucket, 7)}, array_hash{1, hash_for<u64>(step - 1, 6)}, bucket, step);
  }
  stream.sync();
}

#if _CCCL_HAS_INT128()
C2H_TEST("cuco probing accepts public 128-bit hash results", "[probing][hash][cg]")
{
  using hasher = cuda::hash<cuda::std::int32_t, cuda::hash_algorithm::murmurhash3_x64_128>;

  // Reference digests for key 42 from the public MurmurHash3_x64_128 hash tests, with seeds 0 and 42.
  // The high words 0xe2d2... and 0xf9e3... are fractions 0.886 and 0.976 of the range, which scale to
  // bucket 6 of 7 and step index 5 of 6.
  constexpr unsigned first = 6;
  constexpr unsigned step  = 6;

  CHECK(cuco::detail::__hash_to_index(hasher{0}(42), 7) == first);
  CHECK(cuco::detail::__hash_to_index(hasher{42}(42), 6) == step - 1);
  const cuda::stream stream{cuda::device_ref{0}};
  check_probing<1, 1>(stream, hasher{0}, hasher{42}, 42, first, step);
  check_probing<2, 2>(stream, hasher{0}, hasher{42}, 42, first, step);
  stream.sync();
}
#endif // _CCCL_HAS_INT128()

_CCCL_KERNEL_ATTRIBUTES void check_wide_group_offsets_kernel()
{
  constexpr int bucket_size   = 1 << 30;
  constexpr index_type stride = index_type{8} * bucket_size;
  const auto tile             = cg::tiled_partition<8>(cg::this_thread_block());
  const auto lane_offset      = index_type{tile.thread_rank()} * bucket_size;
  const cuda::std::extents<index_type, cuda::std::dynamic_extent> capacity{7 * stride};

  using hasher = constant_hash<index_type>;
  const cuco::linear_probing<8, hasher> linear{hasher{hash_for<index_type>(6, 7)}};
  const cuco::double_hashing<8, hasher> double_hash{
    hasher{hash_for<index_type>(6, 7)}, hasher{hash_for<index_type>(5, 6)}};
  auto linear_iterator = linear.template make_iterator<bucket_size>(tile, 0, capacity);
  auto double_iterator = double_hash.template make_iterator<bucket_size>(tile, 0, capacity);

  // The stride exceeds int, and lanes 4 through 7 have offsets exceeding uint32_t.
  CHECK(*linear_iterator == 6 * stride + lane_offset);
  CHECK(*double_iterator == 6 * stride + lane_offset);
  CHECK(*(++linear_iterator) == lane_offset);
  CHECK(*(++double_iterator) == 5 * stride + lane_offset);
}

C2H_TEST("cuco group probing computes strides and lane offsets at index width", "[probing][overflow][cg]")
{
  const cuda::stream stream{cuda::device_ref{0}};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<8>()), check_wide_group_offsets_kernel);
  stream.sync();
}
