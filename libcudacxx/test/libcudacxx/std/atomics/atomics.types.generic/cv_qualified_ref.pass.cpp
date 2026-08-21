//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// P3323R1: cv-qualified types in atomic and atomic_ref
//
// atomic_ref<const T>:
//   - value_type is remove_cv_t<T> (i.e. T without const)
//   - load, is_lock_free, conversion operator, wait are available
//   - store, exchange, compare_exchange_*, fetch_*, notify_* are NOT available
//
// atomic_ref<volatile T>:
//   - P3323R1 requires is_always_lock_free, but we intentionally allow
//     conditionally-lock-free types like __int128 to remain well-formed
//   - all operations work normally (load, store, exchange, etc.)

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
// UNSUPPORTED: force-tile

#include <cuda/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

// ===== atomic_ref<const T> tests =====

// Helper to check that store is not callable on atomic_ref<const T>.
// We test this by checking that the class simply does not offer the method
// when T is const, courtesy of SFINAE constraints (P3323R1).
template <class AtomicRef>
struct has_store
{
  template <class A>
  TEST_HOST_DEVICE_FUNC static auto test(int)
    -> decltype(cuda::std::declval<const A&>().store(cuda::std::declval<typename A::value_type>()),
                cuda::std::true_type{});
  template <class>
  TEST_HOST_DEVICE_FUNC static auto test(...) -> cuda::std::false_type;

  static constexpr bool value = decltype(test<AtomicRef>(0))::value;
};

template <class AtomicRef>
struct has_exchange
{
  template <class A>
  TEST_HOST_DEVICE_FUNC static auto test(int)
    -> decltype(cuda::std::declval<const A&>().exchange(cuda::std::declval<typename A::value_type>()),
                cuda::std::true_type{});
  template <class>
  TEST_HOST_DEVICE_FUNC static auto test(...) -> cuda::std::false_type;

  static constexpr bool value = decltype(test<AtomicRef>(0))::value;
};

template <class T>
TEST_HOST_DEVICE_FUNC void test_const()
{
  // P3323R1: value_type should be T (the cv-unqualified type), not const T
  using ref_t = cuda::std::atomic_ref<const T>;
  static_assert(cuda::std::is_same_v<typename ref_t::value_type, T>, "value_type should be remove_cv_t<T>");

  T val{};
  const T& cval = val;
  ref_t ref(cval);

  // Read operations must work
  [[maybe_unused]] bool lf     = ref.is_lock_free();
  [[maybe_unused]] T loaded    = ref.load();
  [[maybe_unused]] T converted = static_cast<T>(ref);

  // P3323R1: wait() is allowed, but notify_one()/notify_all() are not.

  // Mutating operations must NOT be available
  static_assert(!has_store<ref_t>::value, "store() should not be available on atomic_ref<const T>");
  static_assert(!has_exchange<ref_t>::value, "exchange() should not be available on atomic_ref<const T>");
}

// ===== atomic_ref<volatile T> tests =====

template <class T, class AtomicRef>
TEST_HOST_DEVICE_FUNC void test_volatile()
{
  // P3323R1: value_type is remove_cv_t<T>
  static_assert(cuda::std::is_same_v<typename AtomicRef::value_type, T>, "value_type should be remove_cv_t<T>");

  // P3323R1: volatile T requires is_always_lock_free
  static_assert(AtomicRef::is_always_lock_free, "test assumes volatile T is lock-free");

  volatile T val{};
  AtomicRef ref(val);

  // Both read and write operations should work on atomic_ref<volatile T>
  ref.store(T(42));
  assert(ref.load() == T(42));
  T prev = ref.exchange(T(7));
  assert(prev == T(42));
  assert(ref.load() == T(7));

  ref.store(T(1), cuda::std::memory_order_release);
  assert(ref.load(cuda::std::memory_order_acquire) == T(1));
  assert(ref.exchange(T(2), cuda::std::memory_order_acq_rel) == T(1));

  T expected = T(2);
  assert(ref.compare_exchange_strong(expected, T(3), cuda::std::memory_order_acq_rel, cuda::std::memory_order_acquire));
  assert(expected == T(2));

  expected = T(2);
  assert(!ref.compare_exchange_strong(expected, T(4), cuda::std::memory_order_seq_cst));
  assert(expected == T(3));

  expected = T(3);
  while (!ref.compare_exchange_weak(expected, T(4), cuda::std::memory_order_relaxed))
  {
    expected = T(3);
  }
  assert(ref.load(cuda::std::memory_order_relaxed) == T(4));
}

#if _CCCL_HAS_INT128()
template <class T>
TEST_HOST_DEVICE_FUNC void test_volatile_int128_extension()
{
  volatile T value{};

  cuda::std::atomic_ref<volatile T> std_ref(value);
  cuda::atomic_ref<volatile T> cuda_ref(value);

  static_assert(cuda::std::is_same_v<typename decltype(std_ref)::value_type, T>);
  static_assert(cuda::std::is_same_v<typename decltype(cuda_ref)::value_type, T>);
}
#endif // _CCCL_HAS_INT128()

int main(int, char**)
{
  // Test atomic_ref<const T>
  test_const<int>();
  test_const<float>();

  // Test atomic_ref<volatile T>
  test_volatile<int, cuda::std::atomic_ref<volatile int>>();
  test_volatile<float, cuda::std::atomic_ref<volatile float>>();
  test_volatile<int, cuda::atomic_ref<volatile int, cuda::thread_scope_system>>();
  test_volatile<int, cuda::atomic_ref<volatile int, cuda::thread_scope_device>>();
  test_volatile<int, cuda::atomic_ref<volatile int, cuda::thread_scope_block>>();

#if _CCCL_HAS_INT128()
  test_volatile_int128_extension<__int128>();
  test_volatile_int128_extension<unsigned __int128>();
#endif // _CCCL_HAS_INT128()

  return 0;
}
