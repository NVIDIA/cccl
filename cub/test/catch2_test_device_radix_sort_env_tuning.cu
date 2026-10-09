// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>

#include <cuda/execution>
#include <cuda/std/execution>
#include <cuda/std/utility>

#include <cstddef>

#include "catch2_radix_sort_env_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"
#include <c2h/vector.h>

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortPairs, device_radix_sort_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortPairsDescending, device_radix_sort_pairs_descending);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortKeys, device_radix_sort_keys);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortKeysDescending, device_radix_sort_keys_descending);

// %PARAM% TEST_LAUNCH lid 0:2

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH != 1

// Radix sort does not accept user-provided functors or iterators, so we cannot use the block_size_extracting_op or
// block_size_extracting_constant_iterator approach. Instead, we verify tuning by measuring allocation sizes: different
// block sizes produce different temporary storage requirements, which is an observable side effect of the tuning being
// applied.
template <typename KeyT, typename ValueT, int BlockThreads>
struct tiny_onesweep_policy_selector
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability cc) const -> cub::RadixSortPolicy
  {
    using default_selector_t               = cub::detail::radix_sort::policy_selector_from_types<KeyT, ValueT, int>;
    auto policy                            = default_selector_t{}(cc);
    policy.algorithm                       = cub::RadixSortAlgorithm::onesweep;
    policy.onesweep.threads_per_block      = BlockThreads;
    policy.onesweep.items_per_thread       = 1;
    policy.single_tile.threads_per_block   = BlockThreads;
    policy.single_tile.items_per_thread    = 1;
    policy.downsweep.threads_per_block     = BlockThreads;
    policy.downsweep.items_per_thread      = 1;
    policy.alt_downsweep.threads_per_block = BlockThreads;
    policy.alt_downsweep.items_per_thread  = 1;
    policy.histogram.private_partitions    = 1;
    return policy;
  }
};

template <typename CallableT, typename PolicySelector>
std::size_t measure_allocated_bytes(CallableT&& run, PolicySelector policy_selector)
{
  auto env              = cuda::execution::tune(policy_selector);
  size_t expected_bytes = 0;
  cuda::std::forward<CallableT>(run)(env, expected_bytes);
  CHECK(expected_bytes > 0);
  return expected_bytes;
}

CUB_TEST_CASE("Device radix sort pairs can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data       = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    auto data_out   = c2h::device_vector<int>(10'000);
    auto values     = c2h::device_vector<int>(10'000);
    auto values_out = c2h::device_vector<int>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortPairs(
        nullptr,
        expected_bytes,
        data.data().get(),
        data_out.data().get(),
        values.data().get(),
        values_out.data().get(),
        static_cast<int>(data.size()),
        0,
        32,
        env));
    device_radix_sort_pairs(
      data.data().get(),
      data_out.data().get(),
      values.data().get(),
      values_out.data().get(),
      static_cast<int>(data.size()),
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs DB can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data       = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    auto data_out   = c2h::device_vector<int>(10'000);
    auto values     = c2h::device_vector<int>(10'000);
    auto values_out = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<int> double_buf(data.data().get(), data_out.data().get());
    cub::DoubleBuffer<int> double_values(values.data().get(), values_out.data().get());
    REQUIRE(cudaSuccess
            == cub::DeviceRadixSort::SortPairs(
              nullptr, expected_bytes, double_buf, double_values, static_cast<int>(data.size()), 0, 32, env));
    device_radix_sort_pairs(
      double_buf,
      double_values,
      static_cast<int>(data.size()),
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs descending can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data       = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    auto data_out   = c2h::device_vector<int>(10'000);
    auto values     = c2h::device_vector<int>(10'000);
    auto values_out = c2h::device_vector<int>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        expected_bytes,
        data.data().get(),
        data_out.data().get(),
        values.data().get(),
        values_out.data().get(),
        static_cast<int>(data.size()),
        0,
        32,
        env));
    device_radix_sort_pairs_descending(
      data.data().get(),
      data_out.data().get(),
      values.data().get(),
      values_out.data().get(),
      static_cast<int>(data.size()),
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs descending DB can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data       = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    auto data_out   = c2h::device_vector<int>(10'000);
    auto values     = c2h::device_vector<int>(10'000);
    auto values_out = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<int> double_buf(data.data().get(), data_out.data().get());
    cub::DoubleBuffer<int> double_values(values.data().get(), values_out.data().get());
    REQUIRE(cudaSuccess
            == cub::DeviceRadixSort::SortPairsDescending(
              nullptr, expected_bytes, double_buf, double_values, static_cast<int>(data.size()), 0, 32, env));
    device_radix_sort_pairs_descending(
      double_buf,
      double_values,
      static_cast<int>(data.size()),
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data     = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    auto data_out = c2h::device_vector<int>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortKeys(
        nullptr, expected_bytes, data.data().get(), data_out.data().get(), static_cast<int>(data.size()), 0, 32, env));
    device_radix_sort_keys(
      data.data().get(),
      data_out.data().get(),
      static_cast<int>(data.size()),
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys DB can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data     = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    auto data_out = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<int> double_buf(data.data().get(), data_out.data().get());
    REQUIRE(cudaSuccess
            == cub::DeviceRadixSort::SortKeys(
              nullptr, expected_bytes, double_buf, static_cast<int>(data.size()), 0, 32, env));
    device_radix_sort_keys(
      double_buf, static_cast<int>(data.size()), 0, 32, stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys descending can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data     = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    auto data_out = c2h::device_vector<int>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortKeysDescending(
        nullptr, expected_bytes, data.data().get(), data_out.data().get(), static_cast<int>(data.size()), 0, 32, env));
    device_radix_sort_keys_descending(
      data.data().get(),
      data_out.data().get(),
      static_cast<int>(data.size()),
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys descending DB can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data     = c2h::device_vector<int>(10'000); // must be larger than the single tile path
    auto data_out = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<int> double_buf(data.data().get(), data_out.data().get());
    REQUIRE(cudaSuccess
            == cub::DeviceRadixSort::SortKeysDescending(
              nullptr, expected_bytes, double_buf, static_cast<int>(data.size()), 0, 32, env));
    device_radix_sort_keys_descending(
      double_buf, static_cast<int>(data.size()), 0, 32, stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys decomposer+bits can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data     = c2h::device_vector<custom_key_t>(10'000);
    auto data_out = c2h::device_vector<custom_key_t>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortKeys(
        nullptr,
        expected_bytes,
        data.data().get(),
        data_out.data().get(),
        static_cast<int>(data.size()),
        keys_decomposer_t{},
        0,
        32,
        env));
    device_radix_sort_keys(
      data.data().get(),
      data_out.data().get(),
      static_cast<int>(data.size()),
      keys_decomposer_t{},
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys decomposer can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data     = c2h::device_vector<custom_key_t>(10'000);
    auto data_out = c2h::device_vector<custom_key_t>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortKeys(
        nullptr,
        expected_bytes,
        data.data().get(),
        data_out.data().get(),
        static_cast<int>(data.size()),
        keys_decomposer_t{},
        env));
    device_radix_sort_keys(
      data.data().get(),
      data_out.data().get(),
      static_cast<int>(data.size()),
      keys_decomposer_t{},
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys DB decomposer can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto buf0 = c2h::device_vector<custom_key_t>(10'000);
    auto buf1 = c2h::device_vector<custom_key_t>(10'000);
    cub::DoubleBuffer<custom_key_t> d_keys(buf0.data().get(), buf1.data().get());
    REQUIRE(cudaSuccess
            == cub::DeviceRadixSort::SortKeys(
              nullptr, expected_bytes, d_keys, static_cast<int>(buf0.size()), keys_decomposer_t{}, env));
    device_radix_sort_keys(
      d_keys,
      static_cast<int>(buf0.size()),
      keys_decomposer_t{},
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys DB decomposer+bits can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto buf0 = c2h::device_vector<custom_key_t>(10'000);
    auto buf1 = c2h::device_vector<custom_key_t>(10'000);
    cub::DoubleBuffer<custom_key_t> d_keys(buf0.data().get(), buf1.data().get());
    REQUIRE(cudaSuccess
            == cub::DeviceRadixSort::SortKeys(
              nullptr, expected_bytes, d_keys, static_cast<int>(buf0.size()), keys_decomposer_t{}, 0, 32, env));
    device_radix_sort_keys(
      d_keys,
      static_cast<int>(buf0.size()),
      keys_decomposer_t{},
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys descending decomposer+bits can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data     = c2h::device_vector<custom_key_t>(10'000);
    auto data_out = c2h::device_vector<custom_key_t>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortKeysDescending(
        nullptr,
        expected_bytes,
        data.data().get(),
        data_out.data().get(),
        static_cast<int>(data.size()),
        keys_decomposer_t{},
        0,
        32,
        env));
    device_radix_sort_keys_descending(
      data.data().get(),
      data_out.data().get(),
      static_cast<int>(data.size()),
      keys_decomposer_t{},
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys descending decomposer can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto data     = c2h::device_vector<custom_key_t>(10'000);
    auto data_out = c2h::device_vector<custom_key_t>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortKeysDescending(
        nullptr,
        expected_bytes,
        data.data().get(),
        data_out.data().get(),
        static_cast<int>(data.size()),
        keys_decomposer_t{},
        env));
    device_radix_sort_keys_descending(
      data.data().get(),
      data_out.data().get(),
      static_cast<int>(data.size()),
      keys_decomposer_t{},
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys descending DB decomposer can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto buf0 = c2h::device_vector<custom_key_t>(10'000);
    auto buf1 = c2h::device_vector<custom_key_t>(10'000);
    cub::DoubleBuffer<custom_key_t> d_keys(buf0.data().get(), buf1.data().get());
    REQUIRE(cudaSuccess
            == cub::DeviceRadixSort::SortKeysDescending(
              nullptr, expected_bytes, d_keys, static_cast<int>(buf0.size()), keys_decomposer_t{}, env));
    device_radix_sort_keys_descending(
      d_keys,
      static_cast<int>(buf0.size()),
      keys_decomposer_t{},
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort keys descending DB decomposer+bits can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto buf0 = c2h::device_vector<custom_key_t>(10'000);
    auto buf1 = c2h::device_vector<custom_key_t>(10'000);
    cub::DoubleBuffer<custom_key_t> d_keys(buf0.data().get(), buf1.data().get());
    REQUIRE(cudaSuccess
            == cub::DeviceRadixSort::SortKeysDescending(
              nullptr, expected_bytes, d_keys, static_cast<int>(buf0.size()), keys_decomposer_t{}, 0, 32, env));
    device_radix_sort_keys_descending(
      d_keys,
      static_cast<int>(buf0.size()),
      keys_decomposer_t{},
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, cub::NullType, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs decomposer+bits can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto kbuf0 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto kbuf1 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vbuf0 = c2h::device_vector<int>(10'000);
    auto vbuf1 = c2h::device_vector<int>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortPairs(
        nullptr,
        expected_bytes,
        kbuf0.data().get(),
        kbuf1.data().get(),
        vbuf0.data().get(),
        vbuf1.data().get(),
        static_cast<int>(kbuf0.size()),
        pairs_decomposer_t{},
        0,
        32,
        env));
    device_radix_sort_pairs(
      kbuf0.data().get(),
      kbuf1.data().get(),
      vbuf0.data().get(),
      vbuf1.data().get(),
      static_cast<int>(kbuf0.size()),
      pairs_decomposer_t{},
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs decomposer can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto keys     = c2h::device_vector<custom_pair_key_t>(10'000);
    auto keys_out = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vals     = c2h::device_vector<int>(10'000);
    auto vals_out = c2h::device_vector<int>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortPairs(
        nullptr,
        expected_bytes,
        keys.data().get(),
        keys_out.data().get(),
        vals.data().get(),
        vals_out.data().get(),
        static_cast<int>(keys.size()),
        pairs_decomposer_t{},
        env));
    device_radix_sort_pairs(
      keys.data().get(),
      keys_out.data().get(),
      vals.data().get(),
      vals_out.data().get(),
      static_cast<int>(keys.size()),
      pairs_decomposer_t{},
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs DB decomposer can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto kbuf0 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto kbuf1 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vbuf0 = c2h::device_vector<int>(10'000);
    auto vbuf1 = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<custom_pair_key_t> d_keys(kbuf0.data().get(), kbuf1.data().get());
    cub::DoubleBuffer<int> d_values(vbuf0.data().get(), vbuf1.data().get());
    REQUIRE(cudaSuccess
            == cub::DeviceRadixSort::SortPairs(
              nullptr, expected_bytes, d_keys, d_values, static_cast<int>(kbuf0.size()), pairs_decomposer_t{}, env));
    device_radix_sort_pairs(
      d_keys,
      d_values,
      static_cast<int>(kbuf0.size()),
      pairs_decomposer_t{},
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs DB decomposer+bits can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto kbuf0 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto kbuf1 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vbuf0 = c2h::device_vector<int>(10'000);
    auto vbuf1 = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<custom_pair_key_t> d_keys(kbuf0.data().get(), kbuf1.data().get());
    cub::DoubleBuffer<int> d_values(vbuf0.data().get(), vbuf1.data().get());
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortPairs(
        nullptr, expected_bytes, d_keys, d_values, static_cast<int>(kbuf0.size()), pairs_decomposer_t{}, 0, 32, env));
    device_radix_sort_pairs(
      d_keys,
      d_values,
      static_cast<int>(kbuf0.size()),
      pairs_decomposer_t{},
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs descending decomposer+bits can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto keys     = c2h::device_vector<custom_pair_key_t>(10'000);
    auto keys_out = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vals     = c2h::device_vector<int>(10'000);
    auto vals_out = c2h::device_vector<int>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        expected_bytes,
        keys.data().get(),
        keys_out.data().get(),
        vals.data().get(),
        vals_out.data().get(),
        static_cast<int>(keys.size()),
        pairs_decomposer_t{},
        0,
        32,
        env));
    device_radix_sort_pairs_descending(
      keys.data().get(),
      keys_out.data().get(),
      vals.data().get(),
      vals_out.data().get(),
      static_cast<int>(keys.size()),
      pairs_decomposer_t{},
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs descending decomposer can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto keys     = c2h::device_vector<custom_pair_key_t>(10'000);
    auto keys_out = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vals     = c2h::device_vector<int>(10'000);
    auto vals_out = c2h::device_vector<int>(10'000);
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        expected_bytes,
        keys.data().get(),
        keys_out.data().get(),
        vals.data().get(),
        vals_out.data().get(),
        static_cast<int>(keys.size()),
        pairs_decomposer_t{},
        env));
    device_radix_sort_pairs_descending(
      keys.data().get(),
      keys_out.data().get(),
      vals.data().get(),
      vals_out.data().get(),
      static_cast<int>(keys.size()),
      pairs_decomposer_t{},
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs descending DB decomposer can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto kbuf0 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto kbuf1 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vbuf0 = c2h::device_vector<int>(10'000);
    auto vbuf1 = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<custom_pair_key_t> d_keys(kbuf0.data().get(), kbuf1.data().get());
    cub::DoubleBuffer<int> d_values(vbuf0.data().get(), vbuf1.data().get());
    REQUIRE(cudaSuccess
            == cub::DeviceRadixSort::SortPairsDescending(
              nullptr, expected_bytes, d_keys, d_values, static_cast<int>(kbuf0.size()), pairs_decomposer_t{}, env));
    device_radix_sort_pairs_descending(
      d_keys,
      d_values,
      static_cast<int>(kbuf0.size()),
      pairs_decomposer_t{},
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

CUB_TEST_CASE("Device radix sort pairs descending DB decomposer+bits can be tuned", "[radix_sort][device]", CUB_SMALL)
{
  auto l = [&](auto env, size_t& expected_bytes) {
    auto kbuf0 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto kbuf1 = c2h::device_vector<custom_pair_key_t>(10'000);
    auto vbuf0 = c2h::device_vector<int>(10'000);
    auto vbuf1 = c2h::device_vector<int>(10'000);
    cub::DoubleBuffer<custom_pair_key_t> d_keys(kbuf0.data().get(), kbuf1.data().get());
    cub::DoubleBuffer<int> d_values(vbuf0.data().get(), vbuf1.data().get());
    REQUIRE(
      cudaSuccess
      == cub::DeviceRadixSort::SortPairsDescending(
        nullptr, expected_bytes, d_keys, d_values, static_cast<int>(kbuf0.size()), pairs_decomposer_t{}, 0, 32, env));
    device_radix_sort_pairs_descending(
      d_keys,
      d_values,
      static_cast<int>(kbuf0.size()),
      pairs_decomposer_t{},
      0,
      32,
      stdexec::env{env, expected_allocation_size(expected_bytes)});
  };

  const auto bytes32  = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 32>{});
  const auto bytes128 = measure_allocated_bytes(l, tiny_onesweep_policy_selector<int, int, 128>{});
  CHECK(bytes32 != bytes128);
}

#endif // TEST_LAUNCH != 1
