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

// constexpr const_reference at(size_type pos) const;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include <stdexcept>

#include "literal.h"
#include <nv/target>

template <class SV>
__host__ void test_at_throw()
{
  using CharT = typename SV::value_type;

  const CharT* str = TEST_STRLIT(CharT, "Hello world!");
  SV sv{str};

  try
  {
    (void) sv.at(12);
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

template <class SV>
__host__ __device__ constexpr void test_at()
{
  using CharT    = typename SV::value_type;
  using SizeT    = typename SV::size_type;
  using ConstRef = typename SV::const_reference;

  static_assert(cuda::std::is_same_v<ConstRef, decltype(SV{}.at(SizeT{}))>);
  static_assert(!noexcept(SV{}.at(SizeT{})));

  const CharT* str = TEST_STRLIT(CharT, "Hello world!");

  SV sv{str};
  assert(sv.at(0) == str[0]);
  assert(sv.at(1) == str[1]);
  assert(sv.at(4) == str[4]);
  assert(sv.at(8) == str[8]);
  assert(sv.at(11) == str[11]);

  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (test_at_throw<SV>();))
  }
}

__host__ __device__ constexpr bool test()
{
  test_at<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_at<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_at<cuda::std::u16string_view>();
  test_at<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_at<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
