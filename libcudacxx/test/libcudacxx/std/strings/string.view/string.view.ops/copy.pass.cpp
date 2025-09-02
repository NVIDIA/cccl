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

// constexpr size_type copy(char_type* s, size_type n, size_type pos = 0) const;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <nv/target>

#include "literal.h"
#include "test_macros.h"

#if TEST_HAS_EXCEPTIONS()
#  include <stdexcept>
#endif // TEST_HAS_EXCEPTIONS()

template <class SV>
__host__ __device__ constexpr void test_copy()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.copy(cuda::std::declval<CharT*>(), SizeT{}))>);
  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.copy(cuda::std::declval<CharT*>(), SizeT{}, SizeT{}))>);

  static_assert(!noexcept(SV{}.copy(cuda::std::declval<CharT*>(), SizeT{})));
  static_assert(!noexcept(SV{}.copy(cuda::std::declval<CharT*>(), SizeT{}, SizeT{})));

  const CharT* str = TEST_STRLIT(CharT, "12345");
  SV sv{str};

  {
    CharT buf[5]{};
    const auto ret = sv.copy(buf, 0);
    assert(ret == 0);
    assert(buf[0] == CharT{});
    assert(buf[1] == CharT{});
    assert(buf[2] == CharT{});
    assert(buf[3] == CharT{});
    assert(buf[4] == CharT{});
  }
  {
    CharT buf[5]{};
    const auto ret = sv.copy(buf, 0, 2);
    assert(ret == 0);
    assert(buf[0] == CharT{});
    assert(buf[1] == CharT{});
    assert(buf[2] == CharT{});
    assert(buf[3] == CharT{});
    assert(buf[4] == CharT{});
  }
  {
    CharT buf[5]{};
    const auto ret = sv.copy(buf, 2);
    assert(ret == 2);
    assert(buf[0] == TEST_CHARLIT(CharT, '1'));
    assert(buf[1] == TEST_CHARLIT(CharT, '2'));
    assert(buf[2] == CharT{});
    assert(buf[3] == CharT{});
    assert(buf[4] == CharT{});
  }
  {
    CharT buf[5]{};
    const auto ret = sv.copy(buf, 2, 3);
    assert(ret == 2);
    assert(buf[0] == TEST_CHARLIT(CharT, '4'));
    assert(buf[1] == TEST_CHARLIT(CharT, '5'));
    assert(buf[2] == CharT{});
    assert(buf[3] == CharT{});
    assert(buf[4] == CharT{});
  }
  {
    CharT buf[5]{};
    const auto ret = sv.copy(buf, 5);
    assert(ret == 5);
    assert(buf[0] == TEST_CHARLIT(CharT, '1'));
    assert(buf[1] == TEST_CHARLIT(CharT, '2'));
    assert(buf[2] == TEST_CHARLIT(CharT, '3'));
    assert(buf[3] == TEST_CHARLIT(CharT, '4'));
    assert(buf[4] == TEST_CHARLIT(CharT, '5'));
  }
  {
    CharT buf[5]{};
    const auto ret = sv.copy(buf, 5, 2);
    assert(ret == 3);
    assert(buf[0] == TEST_CHARLIT(CharT, '3'));
    assert(buf[1] == TEST_CHARLIT(CharT, '4'));
    assert(buf[2] == TEST_CHARLIT(CharT, '5'));
    assert(buf[3] == CharT{});
    assert(buf[4] == CharT{});
  }
}

__host__ __device__ constexpr bool test()
{
  test_copy<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_copy<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_copy<cuda::std::u16string_view>();
  test_copy<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_copy<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

#if TEST_HAS_EXCEPTIONS()
template <class SV>
void test_copy_throw()
{
  using CharT = typename SV::value_type;

  const CharT* str = TEST_STRLIT(CharT, "12345");
  SV sv{str};

  try
  {
    CharT buf[5]{};
    (void) sv.copy(buf, 5, 123098);
    assert(false);
  }
  catch (const ::std::out_of_range&)
  {
    assert(true);
  }
  catch (...)
  {
    assert(false);
  }
}

bool test_exceptions()
{
  test_copy_throw<cuda::std::string_view>();
#  if _CCCL_HAS_CHAR8_T()
  test_copy_throw<cuda::std::u8string_view>();
#  endif // _CCCL_HAS_CHAR8_T()
  test_copy_throw<cuda::std::u16string_view>();
  test_copy_throw<cuda::std::u32string_view>();
#  if _CCCL_HAS_WCHAR_T()
  test_copy_throw<cuda::std::wstring_view>();
#  endif // _CCCL_HAS_WCHAR_T()

  return true;
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test());
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
