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
//   constexpr T bit_ceil(T x) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "test_macros.h"

class A
{};
enum E1 : unsigned char
{
  rEd
};
enum class E2 : unsigned char
{
  red
};

template <typename T>
__host__ __device__ constexpr bool toobig()
{
  return 0 == cuda::std::bit_ceil(cuda::std::numeric_limits<T>::max());
}

int main(int, char**)
{
  //	Make sure we generate a compile-time error for UB
  static_assert(toobig<unsigned char>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is
                                              // not an integral constant expression}}
  static_assert(toobig<unsigned short>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is
                                               // not an integral constant expression}}
  static_assert(toobig<unsigned>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<unsigned long>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is
                                              // not an integral constant expression}}
  static_assert(toobig<unsigned long long>(), ""); // expected-error-re {{{{(static_assert|static assertion)}}
                                                   // expression is not an integral constant expression}}

  static_assert(toobig<uint8_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not an
                                        // integral constant expression}}
  static_assert(toobig<uint16_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<uint32_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<uint64_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<size_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not an
                                       // integral constant expression}}
  static_assert(toobig<uintmax_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                          // an integral constant expression}}
  static_assert(toobig<uintptr_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                          // an integral constant expression}}

  return 0;
}
