//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr string_view operator""sv(const char* str, size_t len) noexcept;

// constexpr u8string_view operator""sv(const char8_t* str, size_t len) noexcept;

// constexpr u16string_view operator""sv(const char16_t* str, size_t len) noexcept;

// constexpr u32string_view operator""sv(const char32_t* str, size_t len) noexcept;

// constexpr wstring_view operator""sv(const wchar_t* str, size_t len) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

_CCCL_DIAG_SUPPRESS_MSVC(4455) // literal suffix identifiers that do not start with an underscore are reserved

#if _CCCL_HAS_CHAR8_T()
using u8string_view = cuda::std::u8string_view;
#else // ^^^ _CCCL_HAS_CHAR8_T() ^^^ / vvv !CCCL_HAS_CHAR8_T() vvv
using u8string_view = cuda::std::string_view;
#endif // ^^^ !_CCCL_HAS_CHAR8_T() ^^^

__host__ __device__ constexpr bool test()
{
  // char

  {
    using namespace cuda::std::literals::string_view_literals;

    static_assert(cuda::std::is_same_v<decltype(""sv), cuda::std::string_view>);
    static_assert(noexcept(operator""sv("", typename cuda::std::string_view::size_type{})));

    assert(""sv.size() == 0);
    assert(" "sv.size() == 1);
    assert("ABC"sv == "ABC");
    assert("ABC"sv == cuda::std::string_view("ABC"));
  }

  // u8string_view

  {
    using namespace cuda::std::literals::string_view_literals;

    static_assert(cuda::std::is_same_v<decltype(u8""sv), u8string_view>);
    static_assert(noexcept(operator""sv(u8"", typename cuda::std::string_view::size_type{})));

    assert(u8""sv.size() == 0);
    assert(u8" "sv.size() == 1);
    assert(u8"ABC"sv == u8"ABC");
    assert(u8"ABC"sv == u8string_view(u8"ABC"));
  }

  // u16string_view

  {
    using namespace cuda::std::literals::string_view_literals;

    static_assert(cuda::std::is_same_v<decltype(u""sv), cuda::std::u16string_view>);
    static_assert(noexcept(operator""sv(u"", typename cuda::std::u16string_view::size_type{})));

    assert(u""sv.size() == 0);
    assert(u" "sv.size() == 1);
    assert(u"ABC"sv == u"ABC");
    assert(u"ABC"sv == cuda::std::u16string_view(u"ABC"));
  }

  // u32string_view

  {
    using namespace cuda::std::literals::string_view_literals;

    static_assert(cuda::std::is_same_v<decltype(U""sv), cuda::std::u32string_view>);
    static_assert(noexcept(operator""sv(U"", typename cuda::std::u32string_view::size_type{})));

    assert(U""sv.size() == 0);
    assert(U" "sv.size() == 1);
    assert(U"ABC"sv == U"ABC");
    assert(U"ABC"sv == cuda::std::u32string_view(U"ABC"));
  }

  // wstring_view

#if _CCCL_HAS_WCHAR_T()
  {
    using namespace cuda::std::literals::string_view_literals;

    static_assert(cuda::std::is_same_v<decltype(L""sv), cuda::std::wstring_view>);
    static_assert(noexcept(operator""sv(L"", typename cuda::std::wstring_view::size_type{})));

    assert(L""sv.size() == 0);
    assert(L" "sv.size() == 1);
    assert(L"ABC"sv == L"ABC");
    assert(L"ABC"sv == cuda::std::wstring_view(L"ABC"));
  }
#endif // _CCCL_HAS_WCHAR_T()

  // test string_view literals in other namespaces
  {
    using namespace cuda::std::string_view_literals;
    assert(""sv.size() == 0);
  }
  {
    using namespace cuda::std::literals;
    assert(""sv.size() == 0);
  }
  {
    using namespace cuda::std;
    assert(""sv.size() == 0);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
