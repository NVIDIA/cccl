//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03

// template <class T>
//   constexpr int popcount(T x) noexcept;

// Returns: The number of bits set to one in the value of x.
//
// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

class A{};
enum       E1 : unsigned char { rEd };
enum class E2 : unsigned char { red };

template <typename T>
__host__ __device__ constexpr bool constexpr_test()
{
	return cuda::std::popcount(T(0)) == 0
	   &&  cuda::std::popcount(T(1)) == 1
	   &&  cuda::std::popcount(T(2)) == 1
	   &&  cuda::std::popcount(T(3)) == 2
	   &&  cuda::std::popcount(T(4)) == 1
	   &&  cuda::std::popcount(T(5)) == 2
	   &&  cuda::std::popcount(T(6)) == 2
	   &&  cuda::std::popcount(T(7)) == 3
	   &&  cuda::std::popcount(T(8)) == 1
	   &&  cuda::std::popcount(T(9)) == 2
	   &&  cuda::std::popcount(cuda::std::numeric_limits<T>::max()) == cuda::std::numeric_limits<T>::digits
	  ;
}


template <typename T>
__host__ __device__ void runtime_test()
{
	ASSERT_SAME_TYPE(int, decltype(cuda::std::popcount(T(0))));
	ASSERT_NOEXCEPT(               cuda::std::popcount(T(0)));

	assert( cuda::std::popcount(T(121)) == 5);
	assert( cuda::std::popcount(T(122)) == 5);
	assert( cuda::std::popcount(T(123)) == 6);
	assert( cuda::std::popcount(T(124)) == 5);
	assert( cuda::std::popcount(T(125)) == 6);
	assert( cuda::std::popcount(T(126)) == 6);
	assert( cuda::std::popcount(T(127)) == 7);
	assert( cuda::std::popcount(T(128)) == 1);
	assert( cuda::std::popcount(T(129)) == 2);
	assert( cuda::std::popcount(T(130)) == 2);
}

int main(int, char **)
{
	static_assert(constexpr_test<unsigned char>(),      "");
	static_assert(constexpr_test<unsigned short>(),     "");
	static_assert(constexpr_test<unsigned>(),           "");
	static_assert(constexpr_test<unsigned long>(),      "");
	static_assert(constexpr_test<unsigned long long>(), "");

	static_assert(constexpr_test<uint8_t>(),   "");
	static_assert(constexpr_test<uint16_t>(),  "");
	static_assert(constexpr_test<uint32_t>(),  "");
	static_assert(constexpr_test<uint64_t>(),  "");
	static_assert(constexpr_test<size_t>(),    "");
	static_assert(constexpr_test<uintmax_t>(), "");
	static_assert(constexpr_test<uintptr_t>(), "");

#ifndef _LIBCUDACXX_HAS_NO_INT128
	static_assert(constexpr_test<__uint128_t>(),        "");
#endif

	runtime_test<unsigned char>();
	runtime_test<unsigned>();
	runtime_test<unsigned short>();
	runtime_test<unsigned long>();
	runtime_test<unsigned long long>();

	runtime_test<uint8_t>();
	runtime_test<uint16_t>();
	runtime_test<uint32_t>();
	runtime_test<uint64_t>();
	runtime_test<size_t>();
	runtime_test<uintmax_t>();
	runtime_test<uintptr_t>();

#ifndef _LIBCUDACXX_HAS_NO_INT128
	runtime_test<__uint128_t>();

	{
	__uint128_t val = 128;

	val <<= 32;
	assert( cuda::std::popcount(val-1) == 39);
	assert( cuda::std::popcount(val)   ==  1);
	assert( cuda::std::popcount(val+1) ==  2);
	val <<= 2;
	assert( cuda::std::popcount(val-1) == 41);
	assert( cuda::std::popcount(val)   ==  1);
	assert( cuda::std::popcount(val+1) ==  2);
	val <<= 3;
	assert( cuda::std::popcount(val-1) == 44);
	assert( cuda::std::popcount(val)   ==  1);
	assert( cuda::std::popcount(val+1) ==  2);
	}
#endif

	return 0;
}
