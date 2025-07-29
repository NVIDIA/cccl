//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr int compare(basic_string_view str) const noexcept;

// constexpr int compare(size_type pos1, size_type n1, basic_string_view str) const;

// constexpr int compare(size_type pos1, size_type n1, basic_string_view str, size_type pos2, size_type n2) const;

// constexpr int compare(const charT* s) const;

// constexpr int compare(size_type pos1, size_type n1, const charT* s) const;

// constexpr int compare(size_type pos1, size_type n1, const charT* s, size_type n2) const;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_compare()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  // constexpr int compare(basic_string_view str) const noexcept;

  static_assert(cuda::std::is_same_v<int, decltype(SV{}.compare(SV{}))>);
#if !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))
  static_assert(noexcept(SV{}.compare(SV{})));
#endif // !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))

  {
    SV sv1{TEST_STRLIT(CharT, "12345")};
    SV sv2{TEST_STRLIT(CharT, "12345")};
    assert(sv1.compare(sv2) == 0);
    assert(sv2.compare(sv1) == 0);
  }
  {
    SV sv1{TEST_STRLIT(CharT, "12345")};
    SV sv2{TEST_STRLIT(CharT, "1234")};
    assert(sv1.compare(sv2) > 0);
    assert(sv2.compare(sv1) < 0);
  }
  {
    SV sv1{TEST_STRLIT(CharT, "12345")};
    SV sv2{TEST_STRLIT(CharT, "12344")};
    assert(sv1.compare(sv2) > 0);
    assert(sv2.compare(sv1) < 0);
  }
  {
    SV sv1{TEST_STRLIT(CharT, "12345")};
    SV sv2{TEST_STRLIT(CharT, "1233")};
    assert(sv1.compare(sv2) > 0);
    assert(sv2.compare(sv1) < 0);
  }

  // constexpr int compare(size_type pos1, size_type n1, basic_string_view str) const;

  static_assert(cuda::std::is_same_v<int, decltype(SV{}.compare(SizeT{}, SizeT{}, SV{}))>);
#if !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))
  static_assert(!noexcept(SV{}.compare(SizeT{}, SizeT{}, SV{})));
#endif // !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))

  {
    SV sv1{TEST_STRLIT(CharT, "12345")};
    SV sv2{TEST_STRLIT(CharT, "2345")};
    assert(sv1.compare(1, 4, sv2) == 0);
  }
  {
    SV sv1{TEST_STRLIT(CharT, "12345")};
    SV sv2{TEST_STRLIT(CharT, "34")};
    assert(sv1.compare(3, 4, sv2) > 0);
    assert(sv2.compare(1, 1, sv1) > 0);
    assert(sv1.compare(0, 4, sv2) < 0);
  }

  // constexpr int compare(size_type pos1, size_type n1, basic_string_view str, size_type pos2, size_type n2) const;

  static_assert(cuda::std::is_same_v<int, decltype(SV{}.compare(SizeT{}, SizeT{}, SV{}, SizeT{}, SizeT{}))>);
#if !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))
  static_assert(!noexcept(SV{}.compare(SizeT{}, SizeT{}, SV{}, SizeT{}, SizeT{})));
#endif // !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))

  {
    SV sv1{TEST_STRLIT(CharT, "12345")};
    SV sv2{TEST_STRLIT(CharT, "12323")};
    assert(sv1.compare(0, 3, sv2, 0, 3) == 0);
    assert(sv1.compare(0, 5, sv2, 0, 3) > 0);
    assert(sv1.compare(1, 2, sv2, 3, 2) == 0);
    assert(sv1.compare(3, 2, sv2, 3, 2) > 0);
  }

  // constexpr int compare(const charT* s) const;

  static_assert(cuda::std::is_same_v<int, decltype(SV{}.compare(cuda::std::declval<const CharT*>()))>);
#if !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))
  static_assert(noexcept(SV{}.compare(cuda::std::declval<const CharT*>())));
#endif // !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))

  {
    SV sv{TEST_STRLIT(CharT, "12345")};
    const CharT* str = TEST_STRLIT(CharT, "12345");
    assert(sv.compare(str) == 0);
  }
  {
    SV sv{TEST_STRLIT(CharT, "12345")};
    const CharT* str = TEST_STRLIT(CharT, "1234");
    assert(sv.compare(str) > 0);
  }
  {
    SV sv{TEST_STRLIT(CharT, "12345")};
    const CharT* str = TEST_STRLIT(CharT, "12344");
    assert(sv.compare(str) > 0);
  }
  {
    SV sv{TEST_STRLIT(CharT, "12345")};
    const CharT* str = TEST_STRLIT(CharT, "1233");
    assert(sv.compare(str) > 0);
  }

  // constexpr int compare(size_type pos1, size_type n1, const charT* s) const;

  static_assert(
    cuda::std::is_same_v<int, decltype(SV{}.compare(SizeT{}, SizeT{}, cuda::std::declval<const CharT*>()))>);
#if !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))
  static_assert(!noexcept(SV{}.compare(SizeT{}, SizeT{}, cuda::std::declval<const CharT*>())));
#endif // !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))

  {
    SV sv{TEST_STRLIT(CharT, "12345")};
    const CharT* str = TEST_STRLIT(CharT, "2345");
    assert(sv.compare(1, 4, str) == 0);
  }
  {
    SV sv{TEST_STRLIT(CharT, "12345")};
    const CharT* str = TEST_STRLIT(CharT, "34");
    assert(sv.compare(3, 4, str) > 0);
    assert(sv.compare(0, 4, str) < 0);
  }

  // constexpr int compare(size_type pos1, size_type n1, const charT* s, size_type n2) const;

  static_assert(
    cuda::std::is_same_v<int, decltype(SV{}.compare(SizeT{}, SizeT{}, cuda::std::declval<const CharT*>(), SizeT{}))>);
#if !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))
  static_assert(!noexcept(SV{}.compare(SizeT{}, SizeT{}, cuda::std::declval<const CharT*>(), SizeT{})));
#endif // !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))

  {
    SV sv{TEST_STRLIT(CharT, "12345")};
    const CharT* str = TEST_STRLIT(CharT, "12323");
    assert(sv.compare(0, 3, str, 0, 3) == 0);
    assert(sv.compare(0, 5, str, 0, 3) > 0);
    assert(sv.compare(1, 2, str, 3, 2) == 0);
    assert(sv.compare(3, 2, str, 3, 2) > 0);
  }
}

__host__ __device__ constexpr bool test()
{
  test_compare<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_compare<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_compare<cuda::std::u16string_view>();
  test_compare<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_compare<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
