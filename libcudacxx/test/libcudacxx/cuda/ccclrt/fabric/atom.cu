//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <testing.cuh>

#if _CCCL_CUDACC_AT_LEAST(13, 4) && __cccl_ptx_isa >= 940

#  include "fabric_test_helper.h"

namespace
{
using namespace fabric_test;

// Atomic targets start at 10, 11, ...; adding 3 distinguishes each old value from its new value.
constexpr int initial_atomic_value = 10;
constexpr int addend               = 3;

// Integer targets follow add, min/max, bitwise operations, exchange, and matching/mismatching CAS.
[[maybe_unused]] constexpr int min_operand       = 9;
constexpr int max_operand                        = 20;
constexpr int and_operand                        = 15;
constexpr int or_operand                         = 32;
constexpr int xor_operand                        = 7;
constexpr int and_result                         = max_operand & and_operand;
constexpr int or_result                          = and_result | or_operand;
[[maybe_unused]] constexpr int xor_result        = or_result ^ xor_operand;
[[maybe_unused]] constexpr int exchange_value    = 50;
constexpr int desired_value                      = 60;
[[maybe_unused]] constexpr int mismatching_value = 999;
[[maybe_unused]] constexpr int unused_desired    = 70;

struct alignas(16) wide_value
{
  cuda::std::uint64_t low;
  cuda::std::uint64_t high;

  // Block load/store must obtain the real address even when operator& is overloaded.
  TEST_FUNC wide_value* operator&()
  {
    return nullptr;
  }

  TEST_FUNC const wide_value* operator&() const
  {
    return nullptr;
  }

  TEST_FUNC friend bool operator==(wide_value lhs, wide_value rhs)
  {
    return lhs.low == rhs.low && lhs.high == rhs.high;
  }
};
static_assert(sizeof(cuda::fabric::atomic_block<cuda::std::uint32_t>) == 16);
static_assert(sizeof(cuda::fabric::atomic_block<cuda::std::uint64_t>) == 16);
static_assert(sizeof(cuda::fabric::atomic_block<wide_value>) == 16);
static_assert(alignof(cuda::fabric::atomic_block<cuda::std::uint32_t>) == 16);
static_assert(sizeof(cuda::fabric::compare_exchange_block<cuda::std::uint32_t>) == 32);
static_assert(sizeof(cuda::fabric::compare_exchange_block<cuda::std::uint64_t>) == 32);
static_assert(offsetof(cuda::fabric::compare_exchange_block<cuda::std::uint32_t>, desired) == 16);
static_assert(offsetof(cuda::fabric::compare_exchange_block<cuda::std::uint64_t>, desired) == 16);
static_assert(offsetof(cuda::fabric::compare_exchange_block<wide_value>, desired) == 16);

template <class T, bool AllSlots = true>
struct atom_kernel : fabric_case<T>
{
  static constexpr cuda::std::size_t stride = AllSlots ? 1 : 16 / sizeof(T);

  TEST_DEVICE_FUNC static T initial_value(cuda::std::size_t i, int = 0)
  {
    // Untargeted elements keep this pattern, detecting writes outside the selected value.
    return T(initial_atomic_value + i / stride);
  }

  TEST_DEVICE_FUNC static T expected_value(cuda::std::size_t i, int = 0)
  {
    bool selected = AllSlots ? i < 16 / sizeof(T) : i % stride == 0;
    return !selected ? initial_value(i)
                     : (cuda::std::is_integral_v<T> ? T{desired_value} : initial_value(i) + T{addend});
  }

  template <class Config>
  TEST_DEVICE_FUNC void operator()(Config, cuda::unicast_logical_endpoint_ref endpoint) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_100, {
      __shared__ cuda::fabric::atomic_block<T> operand;
      __shared__ cuda::fabric::atomic_block<T> old_value;
      __shared__ cuda::fabric::compare_exchange_block<T> cas;
      auto& bar = make_barrier();
      for (cuda::std::uint64_t slot = 0; slot < (AllSlots ? 16 : 64); slot += (AllSlots ? sizeof(T) : 16))
      {
        const cuda::std::uint64_t offset = data_offset + slot;
        T start                          = T(initial_atomic_value + slot / (AllSlots ? sizeof(T) : 16));
        operand.store(offset, T{addend});
        cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
        // Add 3: return start and store start + 3.
        cuda::fabric::try_fetch_add(endpoint, offset, &old_value, &operand, bar);
        cuda::fabric::submit();
        wait_success(bar, 1);
        CHECK(old_value.load(offset) == start);
        CHECK(operand.load(offset) == T{addend});

        if constexpr (cuda::std::is_integral_v<T>)
        {
          operand.store(offset, T{min_operand});
          cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
          // min(start + 3, 9) = 9; return start + 3.
          cuda::fabric::try_fetch_min(endpoint, offset, &old_value, &operand, bar);
          cuda::fabric::submit();
          wait_success(bar, 1);
          CHECK(old_value.load(offset) == start + T{addend});

          operand.store(offset, T{max_operand});
          cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
          // max(9, 20) = 20; return 9.
          cuda::fabric::try_fetch_max(endpoint, offset, &old_value, &operand, bar);
          cuda::fabric::submit();
          wait_success(bar, 1);
          CHECK(old_value.load(offset) == T{min_operand});

          operand.store(offset, T{and_operand});
          cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
          // 20 & 15 = 4; return 20.
          cuda::fabric::try_fetch_and(endpoint, offset, &old_value, &operand, bar);
          cuda::fabric::submit();
          wait_success(bar, 1);
          CHECK(old_value.load(offset) == T{max_operand});

          operand.store(offset, T{or_operand});
          cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
          // 4 | 32 = 36; return 4.
          cuda::fabric::try_fetch_or(endpoint, offset, &old_value, &operand, bar);
          cuda::fabric::submit();
          wait_success(bar, 1);
          CHECK(old_value.load(offset) == T{and_result});

          operand.store(offset, T{xor_operand});
          cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
          // 36 ^ 7 = 35; return 36.
          cuda::fabric::try_fetch_xor(endpoint, offset, &old_value, &operand, bar);
          cuda::fabric::submit();
          wait_success(bar, 1);
          CHECK(old_value.load(offset) == T{or_result});

          operand.store(offset, T{exchange_value});
          cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
          // Replace 35 with 50; return 35.
          cuda::fabric::try_exchange(endpoint, offset, &old_value, &operand, bar);
          cuda::fabric::submit();
          wait_success(bar, 1);
          CHECK(old_value.load(offset) == T{xor_result});

          cas.compare.store(offset, T{exchange_value});
          cas.desired.store(offset, T{desired_value});
          cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
          // 50 matches: store 60 and return 50.
          cuda::fabric::try_compare_exchange(endpoint, offset, &old_value, &cas, bar);
          cuda::fabric::submit();
          wait_success(bar, 1);
          CHECK(old_value.load(offset) == T{exchange_value});
          CHECK(cas.compare.load(offset) == T{exchange_value});
          CHECK(cas.desired.load(offset) == T{desired_value});

          cas.compare.store(offset, T{mismatching_value});
          cas.desired.store(offset, T{unused_desired});
          cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
          // 60 != 999: preserve and return 60.
          cuda::fabric::try_compare_exchange(endpoint, offset, &old_value, &cas, bar);
          cuda::fabric::submit();
          wait_success(bar, 1);
          CHECK(old_value.load(offset) == T{desired_value});
        }
      }
    })
  }
};

template <class T>
struct guarded_atomic_block
{
  cuda::std::uint64_t before[2];
  cuda::fabric::atomic_block<T> value;
  cuda::std::uint64_t after[2];
};

struct multiple_atom_kernel : fabric_case<cuda::std::uint32_t>
{
  static constexpr cuda::std::size_t words_per_block = 16 / sizeof(cuda::std::uint32_t);

  TEST_DEVICE_FUNC static cuda::std::uint32_t initial_value(cuda::std::size_t i, int = 0)
  {
    // Each 16-byte block contains four equal words, starting with 10, 11, 12, and 13.
    return initial_atomic_value + i / words_per_block;
  }

  TEST_DEVICE_FUNC static cuda::std::uint32_t expected_value(cuda::std::size_t i, int = 0)
  {
    return initial_value(i) + (i % words_per_block == 0 ? addend : 0);
  }

  template <class Config>
  TEST_DEVICE_FUNC void operator()(Config, cuda::unicast_logical_endpoint_ref endpoint) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_100, {
      __shared__ cuda::fabric::atomic_block<cuda::std::uint32_t> operands[4];
      __shared__ guarded_atomic_block<cuda::std::uint32_t> old_values[4];
      auto& bar                              = make_barrier();
      constexpr cuda::std::uint64_t sentinel = 0x123456789abcdef0ull;
      for (int i = 0; i < 4; ++i)
      {
        operands[i].store(data_offset + i * 16, addend);
        for (int guard = 0; guard < 2; ++guard)
        {
          old_values[i].before[guard] = old_values[i].after[guard] = sentinel;
        }
      }
      cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
      for (int i = 0; i < 4; ++i)
      {
        // Add 3 to the first word only: return 10 + i and store 13 + i.
        cuda::fabric::try_fetch_add(endpoint, data_offset + i * 16, &old_values[i].value, &operands[i], bar);
      }
      cuda::fabric::submit();
      wait_success(bar, 4);
      for (int i = 0; i < 4; ++i)
      {
        CHECK(old_values[i].value.load(data_offset + i * 16) == cuda::std::uint32_t(initial_atomic_value + i));
        CHECK(operands[i].load(data_offset + i * 16) == cuda::std::uint32_t(addend));
        for (int guard = 0; guard < 2; ++guard)
        {
          CHECK(old_values[i].before[guard] == sentinel);
          CHECK(old_values[i].after[guard] == sentinel);
        }
      }
    })
  }
};

struct wide_atom_kernel : fabric_case<wide_value>
{
  // Target the first 128-bit value; equal guard halves detect writes to any other value.
  static constexpr wide_value initial{10, 20};
  static constexpr wide_value guard{8, 8};
  static constexpr wide_value desired{50, 60};

  TEST_DEVICE_FUNC static wide_value initial_value(cuda::std::size_t i, int = 0)
  {
    if (i == 0)
    {
      return initial;
    }
    return guard;
  }

  TEST_DEVICE_FUNC static wide_value expected_value(cuda::std::size_t i, int = 0)
  {
    return i == 0 ? desired : initial_value(i);
  }

  template <class Config>
  TEST_DEVICE_FUNC void operator()(Config, cuda::unicast_logical_endpoint_ref endpoint) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_100, {
      constexpr auto exchanged      = (wide_value{30, 40});
      constexpr auto mismatching    = (wide_value{50, 99});
      constexpr auto unused_desired = (wide_value{70, 80});
      const auto desired            = expected_value(0);
      __shared__ cuda::fabric::atomic_block<wide_value> operand;
      __shared__ cuda::fabric::atomic_block<wide_value> old_value;
      __shared__ cuda::fabric::compare_exchange_block<wide_value> cas;
      auto& bar = make_barrier();
      operand.store(data_offset, exchanged);
      cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
      // Replace {10, 20} with {30, 40}; return {10, 20}.
      cuda::fabric::try_exchange(endpoint, data_offset, &old_value, &operand, bar);
      cuda::fabric::submit();
      wait_success(bar, 1);
      CHECK(old_value.load(data_offset) == initial);
      CHECK(operand.load(data_offset) == exchanged);

      cas.compare.store(data_offset, exchanged);
      cas.desired.store(data_offset, desired);
      cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
      // Both halves match {30, 40}: store {50, 60} and return {30, 40}.
      cuda::fabric::try_compare_exchange(endpoint, data_offset, &old_value, &cas, bar);
      cuda::fabric::submit();
      wait_success(bar, 1);
      CHECK(old_value.load(data_offset) == exchanged);
      CHECK(cas.compare.load(data_offset) == exchanged);
      CHECK(cas.desired.load(data_offset) == desired);

      cas.compare.store(data_offset, mismatching);
      cas.desired.store(data_offset, unused_desired);
      cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
      // Upper halves differ (60 != 99): preserve and return {50, 60}.
      cuda::fabric::try_compare_exchange(endpoint, data_offset, &old_value, &cas, bar);
      cuda::fabric::submit();
      wait_success(bar, 1);
      CHECK(old_value.load(data_offset) == desired);
    })
  }
};

template <class T>
TEST_DEVICE_FUNC T make_pair(float low, float high);

#  if _CCCL_HAS_NVFP16()
template <>
TEST_DEVICE_FUNC __half2 make_pair(float low, float high)
{
  return __floats2half2_rn(low, high);
}
#  endif

#  if _CCCL_HAS_NVBF16()
template <>
TEST_DEVICE_FUNC __nv_bfloat162 make_pair(float low, float high)
{
  return __floats2bfloat162_rn(low, high);
}
#  endif

template <class T>
struct packed_atom_kernel : fabric_case<T>
{
  TEST_DEVICE_FUNC static T initial_value(cuda::std::size_t, int = 0)
  {
    // Distinct lanes detect accidental scalar handling; untargeted pairs retain this pattern.
    return make_pair<T>(8, 4);
  }

  TEST_DEVICE_FUNC static T expected_value(cuda::std::size_t i, int = 0)
  {
    return i == 0 ? make_pair<T>(12, 7) : initial_value(i);
  }

  template <class Config>
  TEST_DEVICE_FUNC void operator()(Config, cuda::unicast_logical_endpoint_ref endpoint) const
  {
    NV_IF_TARGET(NV_PROVIDES_SM_100, {
      __shared__ cuda::fabric::atomic_block<T> operand;
      __shared__ cuda::fabric::atomic_block<T> old_value;
      auto& bar = make_barrier();

      operand.store(data_offset, make_pair<T>(2, 3));
      cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
      // Lane-wise add: (8, 4) + (2, 3) = (10, 7); return (8, 4).
      cuda::fabric::try_fetch_add(endpoint, data_offset, &old_value, &operand, bar);
      cuda::fabric::submit();
      wait_success(bar, 1);
      CHECK(old_value.load(data_offset) == make_pair<T>(8, 4));
      CHECK(operand.load(data_offset) == make_pair<T>(2, 3));

      operand.store(data_offset, make_pair<T>(6, 9));
      cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
      // Lane-wise min: min((10, 7), (6, 9)) = (6, 7); return (10, 7).
      cuda::fabric::try_fetch_min(endpoint, data_offset, &old_value, &operand, bar);
      cuda::fabric::submit();
      wait_success(bar, 1);
      CHECK(old_value.load(data_offset) == make_pair<T>(10, 7));
      CHECK(operand.load(data_offset) == make_pair<T>(6, 9));

      operand.store(data_offset, make_pair<T>(12, 5));
      cuda::fence_proxy_async<cuda::proxy_async_space::shared_cta>();
      // Lane-wise max: max((6, 7), (12, 5)) = (12, 7); return (6, 7).
      cuda::fabric::try_fetch_max(endpoint, data_offset, &old_value, &operand, bar);
      cuda::fabric::submit();
      wait_success(bar, 1);
      CHECK(old_value.load(data_offset) == make_pair<T>(6, 7));
      CHECK(operand.load(data_offset) == make_pair<T>(12, 5));
    })
  }
};

template <class T, bool AllSlots = true>
void test_atom()
{
  INFO("atomic element size=" << sizeof(T) << ", integral=" << cuda::std::is_integral_v<T>);
  run_unicast(atom_kernel<T, AllSlots>{});
}
} // namespace

C2H_CCCLRT_TEST("direct fabric atom staging blocks at 16-byte-aligned endpoint offsets", "[fabric]")
{
  test_atom<cuda::std::uint32_t, false>();
  test_atom<cuda::std::uint64_t, false>();
  test_atom<float, false>();
  test_atom<double, false>();
}

// TODO: Add nonzero-slot coverage in a separate PR after the fabric.try_atom
// compiler lowering bug is fixed. For now, test only 16-byte-aligned offsets.

C2H_CCCLRT_TEST("direct fabric multiple atoms share one barrier phase and preserve shared guards", "[fabric]")
{
  run_unicast(multiple_atom_kernel{});
}

C2H_CCCLRT_TEST("direct fabric 128-bit exchange and compare-exchange", "[fabric]")
{
  wide_value value{};
  const auto& const_value = value;
  CHECK(&value == nullptr);
  CHECK(&const_value == nullptr);
  run_unicast(wide_atom_kernel{});
}

C2H_CCCLRT_TEST("direct fabric packed half and bfloat16 atomic operations", "[fabric]")
{
#  if _CCCL_HAS_NVFP16()
  run_unicast(packed_atom_kernel<__half2>{});
#  endif
#  if _CCCL_HAS_NVBF16()
  run_unicast(packed_atom_kernel<__nv_bfloat162>{});
#  endif
}

#else
C2H_TEST("direct fabric atom runtime tests require CUDA 13.4", "[fabric]")
{
  SUCCEED("fabric operations and shared_barrier require CUDA 13.4");
}
#endif
