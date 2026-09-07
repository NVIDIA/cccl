//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr basic_string_view subview(size_type pos = 0, size_type n = npos) const;

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
TEST_FUNC constexpr void test_subview()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_same_v<SV, decltype(SV{}.subview())>);
  static_assert(cuda::std::is_same_v<SV, decltype(SV{}.subview(SizeT{}))>);
  static_assert(cuda::std::is_same_v<SV, decltype(SV{}.subview(SizeT{}, SizeT{}))>);

#if !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))
  static_assert(!noexcept(SV{}.subview()));
  static_assert(!noexcept(SV{}.subview(SizeT{})));
  static_assert(!noexcept(SV{}.subview(SizeT{}, SizeT{})));
#endif // !(_CCCL_COMPILER(GCC, <, 9) || _CCCL_COMPILER(MSVC))

  const CharT* str = TEST_STRLIT(CharT, "12345");
  SV sv{str};

  {
    const auto sub_sv = sv.subview();
    assert(sub_sv.data() == sv.data());
    assert(sub_sv.size() == sv.size());
  }
  {
    const auto sub_sv = sv.subview(0);
    assert(sub_sv.data() == sv.data());
    assert(sub_sv.size() == sv.size());
  }
  {
    const auto sub_sv = sv.subview(2);
    assert(sub_sv.data() == sv.data() + 2);
    assert(sub_sv.size() == 3);
  }
  {
    const auto sub_sv = sv.subview(3, 2);
    assert(sub_sv.data() == sv.data() + 3);
    assert(sub_sv.size() == 2);
  }
  {
    const auto sub_sv = sv.subview(5);
    assert(sub_sv.data() == sv.data() + 5);
    assert(sub_sv.size() == 0);
  }
  {
    const auto sub_sv = sv.subview(5, 2);
    assert(sub_sv.data() == sv.data() + 5);
    assert(sub_sv.size() == 0);
  }
}

TEST_FUNC constexpr bool test()
{
  test_subview<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_subview<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_subview<cuda::std::u16string_view>();
  test_subview<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_subview<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

#if TEST_HAS_EXCEPTIONS()
template <class SV>
void test_subview_throw()
{
  using CharT = typename SV::value_type;

  const CharT* str = TEST_STRLIT(CharT, "12345");
  SV sv{str};

  try
  {
    (void) sv.subview(123098);
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
  test_subview_throw<cuda::std::string_view>();
#  if _CCCL_HAS_CHAR8_T()
  test_subview_throw<cuda::std::u8string_view>();
#  endif // _CCCL_HAS_CHAR8_T()
  test_subview_throw<cuda::std::u16string_view>();
  test_subview_throw<cuda::std::u32string_view>();
#  if _CCCL_HAS_WCHAR_T()
  test_subview_throw<cuda::std::wstring_view>();
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
