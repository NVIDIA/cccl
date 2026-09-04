//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// UNSUPPORTED: force-tile
// error: asm statement is unsupported in tile code

// <cuda/std/atomic>

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename T>
TEST_HOST_DEVICE_FUNC void check_supported_type(T v)
{
  cuda::std::atomic<T> atom(v);
  cuda::std::atomic_ref<T> ref(v);
}

template <size_t N>
struct sized_type
{
  char data[N];

  TEST_HOST_DEVICE_FUNC bool operator==(const sized_type& other) const
  {
    for (size_t i = 0; i < N; ++i)
    {
      if (data[i] != other.data[i])
      {
        return false;
      }
    }
    return true;
  }
  TEST_HOST_DEVICE_FUNC bool operator!=(const sized_type& other) const
  {
    return !(*this == other);
  }
};

using size3_t = sized_type<3>;
using size5_t = sized_type<5>;
using size6_t = sized_type<6>;
using size7_t = sized_type<7>;

// Owned atomics widen every size below 8 bytes to the next power of two, so all of them are
// lock-free. atomic_ref works on the raw object and can only use the hardware supported
// power-of-two widths, so non-power-of-two sizes are not lock-free there.
static_assert(cuda::std::atomic<size3_t>::is_always_lock_free, "3-byte atomic widens to 4 bytes");
static_assert(cuda::std::atomic<size5_t>::is_always_lock_free, "5-byte atomic widens to 8 bytes");
static_assert(cuda::std::atomic<size6_t>::is_always_lock_free, "6-byte atomic widens to 8 bytes");
static_assert(cuda::std::atomic<size7_t>::is_always_lock_free, "7-byte atomic widens to 8 bytes");
static_assert(sizeof(cuda::std::atomic<size3_t>) == 4, "3-byte atomic widens to 4 bytes of storage");
static_assert(sizeof(cuda::std::atomic<size5_t>) == 8, "5-byte atomic widens to 8 bytes of storage");
static_assert(sizeof(cuda::std::atomic<size6_t>) == 8, "6-byte atomic widens to 8 bytes of storage");
static_assert(sizeof(cuda::std::atomic<size7_t>) == 8, "7-byte atomic widens to 8 bytes of storage");
static_assert(!cuda::std::atomic_ref<size3_t>::is_always_lock_free, "3-byte atomic_ref is not lock-free");
static_assert(!cuda::std::atomic_ref<size5_t>::is_always_lock_free, "5-byte atomic_ref is not lock-free");
static_assert(!cuda::std::atomic_ref<size6_t>::is_always_lock_free, "6-byte atomic_ref is not lock-free");
static_assert(!cuda::std::atomic_ref<size7_t>::is_always_lock_free, "7-byte atomic_ref is not lock-free");

template <typename T>
TEST_HOST_DEVICE_FUNC void check_roundtrip(T v)
{
  cuda::std::atomic<T> atom(v);
  assert(atom.load() == v);
  atom.store(T{});
  assert(atom.load() == T{});
  atom.exchange(v);
  assert(atom.load() == v);
  T expected = v;
  assert(atom.compare_exchange_strong(expected, T{}));
  assert(atom.load() == T{});
}

int main(int, char**)
{
  check_supported_type(size3_t{});
  check_supported_type(size5_t{});
  check_supported_type(size6_t{});
  check_supported_type(size7_t{});

  check_roundtrip(size3_t{{1, 2, 3}});
  check_roundtrip(size5_t{{1, 2, 3, 4, 5}});
  check_roundtrip(size6_t{{1, 2, 3, 4, 5, 6}});
  check_roundtrip(size7_t{{1, 2, 3, 4, 5, 6, 7}});

  check_supported_type(static_cast<char>(0));
  check_supported_type(static_cast<signed char>(0));
  check_supported_type(static_cast<unsigned char>(0));
  check_supported_type(static_cast<short>(0));
  check_supported_type(static_cast<unsigned short>(0));
  check_supported_type(static_cast<int>(0));
  check_supported_type(static_cast<unsigned int>(0));
  check_supported_type(static_cast<long>(0));
  check_supported_type(static_cast<unsigned long>(0));
  check_supported_type(static_cast<long long>(0));
  check_supported_type(static_cast<unsigned long long>(0));
  check_supported_type(static_cast<wchar_t>(0));
  check_supported_type(static_cast<char16_t>(0));
  check_supported_type(static_cast<char32_t>(0));
  check_supported_type(static_cast<uintptr_t>(0));
  check_supported_type(static_cast<uint8_t>(0));
  check_supported_type(static_cast<int16_t>(0));
  check_supported_type(static_cast<uint16_t>(0));
  check_supported_type(static_cast<int32_t>(0));
  check_supported_type(static_cast<uint32_t>(0));
  check_supported_type(static_cast<int64_t>(0));
  check_supported_type(static_cast<uint64_t>(0));
#if _CCCL_HAS_INT128()
  NV_IF_TARGET(NV_IS_DEVICE,
               // Perform check only on device
               (check_supported_type(static_cast<__int128_t>(0)); check_supported_type(static_cast<__uint128_t>(0));))
#endif // _CCCL_HAS_INT128()

  return 0;
}
