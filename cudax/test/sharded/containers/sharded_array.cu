//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief `sharded_array` container tests: allocation (specs, place_group,
 *        uniform), host roundtrips, copy_between/resharding, allocate_like,
 *        adoption, slicing, validation and the contract throws.
 *
 * Runs on any machine with at least one GPU; multi-place coverage uses the
 * locality domains of device 0 (or the whole device where unsupported).
 */

#include <cuda/experimental/sharded.cuh>

#include <numeric>
#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
template <typename T>
::std::vector<T> sequential(size_t n)
{
  ::std::vector<T> data(n);
  ::std::iota(data.begin(), data.end(), T(1)); // 1, 2, 3, ...
  return data;
}

template <typename T>
void expect_equal(const ::std::vector<T>& actual, const ::std::vector<T>& expected)
{
  EXPECT(actual.size() == expected.size());
  for (size_t i = 0; i < actual.size(); i++)
  {
    EXPECT(actual[i] == expected[i]);
  }
}

void test_single_device_roundtrip()
{
  const size_t n = 1000;
  auto input     = sequential<unsigned long long>(n);

  auto arr = sharded_array<unsigned long long>::allocate({{n, data_place::device(0), exec_place::device(0), nullptr}});
  EXPECT(arr.num_shards() == 1UL);
  EXPECT(arr.size() == n);
  EXPECT(arr.is_owning());
  EXPECT(arr.validate());

  arr.copy_from_host(input.data());

  ::std::vector<unsigned long long> output(n);
  arr.copy_to_host(output.data());
  expect_equal(output, input);
}

void test_multi_shard_roundtrip()
{
  const size_t n = 1000;
  auto input     = sequential<unsigned long long>(n);

  ::std::vector<shard_spec> specs;
  for (int i = 0; i < 4; i++)
  {
    specs.emplace_back(250, data_place::device(0), exec_place::device(0), nullptr);
  }
  auto arr = sharded_array<unsigned long long>::allocate(specs);
  EXPECT(arr.num_shards() == 4UL);
  EXPECT(arr.size() == n);
  EXPECT(arr.validate());

  // Shard index math
  EXPECT(arr.shard(1).global_offset == 250UL);
  EXPECT(arr.shard(1).contains(300));
  EXPECT(!arr.shard(1).contains(500));
  EXPECT(arr.shard(1).to_local(300) == 50UL);
  EXPECT(arr.shard(1).to_global(50) == 300UL);
  EXPECT(arr.shard(3).global_end() == n);

  arr.copy_from_host(input.data());
  ::std::vector<unsigned long long> output(n);
  arr.copy_to_host(output.data());
  expect_equal(output, input);
}

void test_place_group_allocation()
{
  auto group     = place_group{make_locality_domain_grid(0)};
  const size_t n = 10000;

  auto arr = sharded_array<long long>::allocate(group, n);
  EXPECT(arr.num_shards() == group.size()); // one shard per place, empty or not
  EXPECT(arr.size() == n);
  EXPECT(arr.is_owning());
  EXPECT(arr.validate());

  // Reference streams come from the group
  for (size_t i = 0; i < arr.num_shards(); i++)
  {
    EXPECT(arr.shard(i).stream != nullptr);
  }

  auto input = sequential<long long>(n);
  arr.copy_from_host(input.data());
  ::std::vector<long long> output(n);
  arr.copy_to_host(output.data());
  expect_equal(output, input);

  // Explicit per-shard sizes must match the group's place count
  bool threw = false;
  try
  {
    auto bad = sharded_array<long long>::allocate(group, ::std::vector<size_t>(group.size() + 1, 10));
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
}

void test_allocate_like()
{
  auto group = place_group{make_locality_domain_grid(0)};
  auto src   = sharded_array<long long>::allocate(group, 999);

  auto dst = sharded_array<long long>::allocate_like(src);
  EXPECT(dst.num_shards() == src.num_shards());
  EXPECT(dst.size() == src.size());
  for (size_t i = 0; i < src.num_shards(); i++)
  {
    EXPECT(dst.shard(i).size == src.shard(i).size);
    EXPECT(dst.shard(i).global_offset == src.shard(i).global_offset);
    EXPECT(dst.shard(i).place == src.shard(i).place);
    EXPECT(dst.shard(i).exec == src.shard(i).exec);
  }

  // Different element type, same layout
  auto dst_f = sharded_array<float>::allocate_like(src);
  EXPECT(dst_f.num_shards() == src.num_shards());
  EXPECT(dst_f.size() == src.size());
}

void test_copy_between_resharding()
{
  const size_t n = 1000;
  auto input     = sequential<unsigned long long>(n);

  auto one = sharded_array<unsigned long long>::allocate({{n, data_place::device(0), exec_place::device(0), nullptr}});
  one.copy_from_host(input.data());

  // 1 shard -> 2 shards
  auto two = sharded_array<unsigned long long>::allocate(
    {{500, data_place::device(0), exec_place::device(0), nullptr},
     {500, data_place::device(0), exec_place::device(0), nullptr}});
  copy_between(one, two);
  ::std::vector<unsigned long long> output(n);
  two.copy_to_host(output.data());
  expect_equal(output, input);

  // misaligned: 3 shards (333+333+334) -> 2 shards (500+500)
  auto three = sharded_array<unsigned long long>::allocate(
    {{333, data_place::device(0), exec_place::device(0), nullptr},
     {333, data_place::device(0), exec_place::device(0), nullptr},
     {334, data_place::device(0), exec_place::device(0), nullptr}});
  three.copy_from_host(input.data());

  auto dst = sharded_array<unsigned long long>::allocate(
    {{500, data_place::device(0), exec_place::device(0), nullptr},
     {500, data_place::device(0), exec_place::device(0), nullptr}});
  copy_between(three, dst);
  dst.copy_to_host(output.data());
  expect_equal(output, input);

  // 2 shards -> 1 shard
  auto back = sharded_array<unsigned long long>::allocate({{n, data_place::device(0), exec_place::device(0), nullptr}});
  copy_between(dst, back);
  back.copy_to_host(output.data());
  expect_equal(output, input);
}

void test_adoption_and_slice()
{
  const size_t n = 700;
  auto input     = sequential<long long>(n);

  auto group = place_group{make_locality_domain_grid(0)};
  auto owner = sharded_array<long long>::allocate(group, n);
  owner.copy_from_host(input.data());

  // Adopt the owner's shards as a non-owning view
  ::std::vector<shard<long long>> shards(owner.begin(), owner.end());
  sharded_array<long long> view(::std::move(shards));
  EXPECT(view.is_view());
  EXPECT(view.size() == n);
  ::std::vector<long long> output(n);
  view.copy_to_host(output.data());
  expect_equal(output, input);

  // Slice [100, 400): values 101..400
  auto sliced = view.slice(100, 400);
  EXPECT(sliced.is_view());
  EXPECT(sliced.size() == 300UL);
  EXPECT(sliced.num_shards() == view.num_shards()); // place correspondence preserved
  EXPECT(sliced.validate());
  ::std::vector<long long> sliced_host(300);
  sliced.copy_to_host(sliced_host.data());
  ::std::vector<long long> sliced_ref(input.begin() + 100, input.begin() + 400);
  expect_equal(sliced_host, sliced_ref);

  // A slice past a shard keeps an empty shard in its position
  auto tail = view.slice(n - 1);
  EXPECT(tail.size() == 1UL);
  EXPECT(tail.num_shards() == view.num_shards());

  // adopt() is the named form of the adopting constructor: same zero-copy
  // view semantics, same data identity (the owner's pointers, unchanged).
  ::std::vector<shard<long long>> shards2(owner.begin(), owner.end());
  auto adopted = sharded_array<long long>::adopt(::std::move(shards2));
  EXPECT(adopted.is_view());
  EXPECT(!adopted.is_owning());
  EXPECT(adopted.size() == n);
  EXPECT(adopted.num_shards() == view.num_shards());
  for (size_t i = 0; i < adopted.num_shards(); ++i)
  {
    EXPECT(adopted[i].data == view[i].data);
    EXPECT(adopted[i].size == view[i].size);
  }
}

// Co-partition validation now lives in the concept tier and is exercised
// through the generic algorithms (was: the container-tier check_compatible).
void test_copartition_refusal()
{
  auto a = sharded_array<long long>::allocate({{100, data_place::device(0), exec_place::device(0), nullptr}});
  auto b = sharded_array<long long>::allocate(
    {{50, data_place::device(0), exec_place::device(0), nullptr},
     {50, data_place::device(0), exec_place::device(0), nullptr}});
  auto c = sharded_array<long long>::allocate({{60, data_place::device(0), exec_place::device(0), nullptr}});

  const auto identity = [] __device__(long long x) {
    return x;
  };

  bool threw = false;
  try
  {
    zip_transform(a, identity, b);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw); // shard count mismatch refused before any launch

  threw = false;
  try
  {
    zip_transform(a, identity, c);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw); // shard region mismatch refused before any launch

  zip_transform(a, identity, a); // co-partitioned with itself: must not throw
  cuda_safe_call(cudaDeviceSynchronize());
}

void test_uniform_and_host()
{
  const size_t n = 1001;
  auto input     = sequential<long long>(n);

  auto uni = sharded_array<long long>::allocate_uniform(n, {0});
  EXPECT(uni.size() == n);
  uni.copy_from_host(input.data());
  ::std::vector<long long> output(n);
  uni.copy_to_host(output.data());
  expect_equal(output, input);

  auto host = sharded_array<long long>::allocate_host(n);
  EXPECT(host.size() == n);
  host.copy_from_host(input.data());
  host.copy_to_host(output.data());
  expect_equal(output, input);

  auto fh = sharded_array<long long>::from_host(input.data(), {{n, data_place::device(0)}});
  fh.copy_to_host(output.data());
  expect_equal(output, input);
}
} // namespace

void test_empty_shard_allocation()
{
  auto group = place_group{make_locality_domain_grid()};
  if (group.size() < 2)
  {
    return; // needs at least two places so one can be empty
  }

  // One place gets a zero size: the shard EXISTS (place correspondence is
  // preserved), it just holds no storage.
  ::std::vector<size_t> sizes(group.size(), 0);
  const size_t n = 1000;
  sizes[0]       = n;

  auto arr = sharded_array<long long>::allocate(group, sizes);
  EXPECT(arr.num_shards() == group.size());
  EXPECT(arr.size() == n);
  EXPECT(arr.shard(1).size == 0);
  EXPECT(arr.shard(1).data == nullptr);
  EXPECT(arr.shard(1).capacity == 0);
  EXPECT(arr.validate());

  // Host round-trip crosses the empty shard unharmed
  auto input = sequential<long long>(n);
  arr.copy_from_host(input.data());
  ::std::vector<long long> output(n);
  arr.copy_to_host(output.data());
  expect_equal(output, input);
}

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  test_single_device_roundtrip();
  test_multi_shard_roundtrip();
  test_place_group_allocation();
  test_allocate_like();
  test_copy_between_resharding();
  test_adoption_and_slice();
  test_copartition_refusal();
  test_uniform_and_host();
  test_empty_shard_allocation();

  return 0;
}
