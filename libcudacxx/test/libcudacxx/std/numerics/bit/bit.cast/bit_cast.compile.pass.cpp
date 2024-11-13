//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/bit>
//
// template<class To, class From>
//   constexpr To bit_cast(const From& from) noexcept;

// This test makes sure that std::bit_cast fails when any of the following
// constraints are violated:
//
//      (1.1) sizeof(To) == sizeof(From) is true;
//      (1.2) is_trivially_copyable_v<To> is true;
//      (1.3) is_trivially_copyable_v<From> is true.
//
// Also check that it's ill-formed when the return type would be
// ill-formed, even though that is not explicitly mentioned in the
// specification (but it can be inferred from the synopsis).

#include <cuda/std/bit>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

template <class To, class From, class = void>
struct bit_cast_is_valid : cuda::std::false_type
{};

template <class To, class From>
struct bit_cast_is_valid<To, From, decltype(cuda::std::bit_cast<To>(cuda::std::declval<const From&>()))>
    : cuda::std::is_same<To, decltype(cuda::std::bit_cast<To>(cuda::std::declval<const From&>()))>
{};

// Types are not the same size
namespace ns1
{
struct To
{
  char a;
};
struct From
{
  char a;
  char b;
};
static_assert(!bit_cast_is_valid<To, From>::value, "");
static_assert(!bit_cast_is_valid<From&, From>::value, "");
} // namespace ns1

// To is not trivially copyable
namespace ns2
{
struct To
{
  char a;
  __host__ __device__ To(To const&);
};
struct From
{
  char a;
};
static_assert(!bit_cast_is_valid<To, From>::value, "");
} // namespace ns2

// From is not trivially copyable
namespace ns3
{
struct To
{
  char a;
};
struct From
{
  char a;
  __host__ __device__ From(From const&);
};
static_assert(!bit_cast_is_valid<To, From>::value, "");
} // namespace ns3

// The return type is ill-formed
namespace ns4
{
struct From
{
  char a;
  char b;
};
static_assert(!bit_cast_is_valid<char[2], From>::value, "");
static_assert(!bit_cast_is_valid<int(), From>::value, "");
} // namespace ns4

int main(int, char**)
{
  return 0;
}
