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

//  template <class Range>
//  constexpr basic_string_view(Range&& range);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"
#include "test_iterators.h"
#include "test_range.h"

template <class CharT>
__host__ __device__ constexpr void test_from_range()
{
  using SV = cuda::std::basic_string_view<CharT>;

  constexpr auto range_data = TEST_STRLIT(CharT, "range_data");
  constexpr auto conv_data  = TEST_STRLIT(CharT, "conv_data");

  // 1. test construction from cuda::std::array
  {
    cuda::std::array<CharT, 4> arr{range_data[0], range_data[1], range_data[2], range_data[3]};
    auto sv = SV(arr);

    static_assert(cuda::std::is_same_v<decltype(sv), SV>);
    assert(sv.size() == arr.size());
    assert(sv.data() == arr.data());
  }

  // 2. test construction from a type with a non-const conversion operator
  {
    struct NonConstConversionOperator
    {
      __host__ __device__ constexpr const CharT* begin() const
      {
        return range_data;
      }
      __host__ __device__ constexpr const CharT* end() const
      {
        return range_data + 4;
      }
      __host__ __device__ constexpr operator SV()
      {
        return conv_data;
      }
    };

    static_assert(cuda::std::is_constructible_v<SV, NonConstConversionOperator>);
    static_assert(!cuda::std::is_constructible_v<SV, const NonConstConversionOperator&>);

    NonConstConversionOperator nc{};
    SV sv = NonConstConversionOperator{};
    assert(sv == conv_data);
  }

  // 3. test construction from a type with a const conversion operator
  {
    struct ConstConversionOperator
    {
      __host__ __device__ constexpr const CharT* begin() const
      {
        return range_data;
      }
      __host__ __device__ constexpr const CharT* end() const
      {
        return range_data + 4;
      }
      __host__ __device__ constexpr operator SV() const
      {
        return conv_data;
      }
    };

    static_assert(cuda::std::is_constructible_v<SV, ConstConversionOperator>);
    static_assert(cuda::std::is_constructible_v<SV, const ConstConversionOperator&>);

    {
      ConstConversionOperator cv{};
      SV sv = cv;
      assert(sv == conv_data);
    }
    {
      const ConstConversionOperator cv{};
      SV sv = cv;
      assert(sv == conv_data);
    }
  }

  // 4. test construction from a type with a deleted conversion operator
  {
    struct DeletedConversionOperator
    {
      __host__ __device__ constexpr const CharT* begin() const
      {
        return range_data;
      }
      __host__ __device__ constexpr const CharT* end() const
      {
        return range_data + 4;
      }
      operator SV() = delete;
    };

    static_assert(cuda::std::is_constructible_v<SV, DeletedConversionOperator>);
    static_assert(cuda::std::is_constructible_v<SV, const DeletedConversionOperator>);

    {
      DeletedConversionOperator cv{};
      SV sv = cv;
      assert(sv == range_data);
    }
    {
      const DeletedConversionOperator cv{};
      SV sv = cv;
      assert(sv == range_data);
    }
  }

  // 5. test construction from a type with a deleted const conversion operator
  {
    struct DeletedConstConversionOperator
    {
      __host__ __device__ constexpr const CharT* begin() const
      {
        return range_data;
      }
      __host__ __device__ constexpr const CharT* end() const
      {
        return range_data + 4;
      }
      operator SV() const = delete;
    };

    static_assert(cuda::std::is_constructible_v<SV, DeletedConstConversionOperator>);
    static_assert(cuda::std::is_constructible_v<SV, const DeletedConstConversionOperator>);

    {
      DeletedConstConversionOperator cv{};
      SV sv = cv;
      assert(sv == range_data);
    }
    {
      const DeletedConstConversionOperator cv{};
      SV sv = cv;
      assert(sv == range_data);
    }
  }

  // 6. test construction from a string_view with different traits
  {
    struct OtherTraits : cuda::std::char_traits<CharT>
    {};
    cuda::std::basic_string_view<CharT> sv1{TEST_STRLIT(CharT, "hello")};
    cuda::std::basic_string_view<CharT, OtherTraits> sv2(sv1);
    assert(sv1.size() == sv2.size());
    assert(sv1.data() == sv2.data());
  }

  // 7. test construction from L- and R-values
  {
    static_assert(cuda::std::is_constructible_v<SV, cuda::std::array<CharT, 10>&>);
    static_assert(cuda::std::is_constructible_v<SV, const cuda::std::array<CharT, 10>&>);
    static_assert(cuda::std::is_constructible_v<SV, cuda::std::array<CharT, 10>&&>);
    static_assert(cuda::std::is_constructible_v<SV, const cuda::std::array<CharT, 10>&&>);
  }

  // 8. test construction from a sized but not contiguous range
  {
    using SizedButNotContiguousRange = cuda::std::ranges::subrange<random_access_iterator<CharT*>>;
    static_assert(!cuda::std::ranges::contiguous_range<SizedButNotContiguousRange>);
    static_assert(cuda::std::ranges::sized_range<SizedButNotContiguousRange>);
    static_assert(!cuda::std::is_constructible_v<SV, SizedButNotContiguousRange>);
  }

  // 9. test construction from a contiguous but not sized range
  {
    using ContiguousButNotSizedRange =
      cuda::std::ranges::subrange<contiguous_iterator<CharT*>,
                                  sentinel_wrapper<contiguous_iterator<CharT*>>,
                                  cuda::std::ranges::subrange_kind::unsized>;
    static_assert(cuda::std::ranges::contiguous_range<ContiguousButNotSizedRange>);
    static_assert(!cuda::std::ranges::sized_range<ContiguousButNotSizedRange>);
    static_assert(!cuda::std::is_constructible_v<SV, ContiguousButNotSizedRange>);
  }

  // 10. test construction from a range with a different value_type
  {
    using DifferentCharT = cuda::std::conditional_t<cuda::std::is_same_v<CharT, char16_t>, char, char16_t>;
    static_assert(!cuda::std::is_constructible_v<SV, DifferentCharT>);
  }

  // 11. test construction from a L- and R-values with conversion operators
  {
    struct WithStringViewConversionOperator
    {
      __host__ __device__ constexpr const CharT* begin() const;
      __host__ __device__ constexpr const CharT* end() const;
      __host__ __device__ constexpr operator SV() const
      {
        return {};
      }
    };

    static_assert(cuda::std::is_constructible_v<SV, WithStringViewConversionOperator>);
    static_assert(cuda::std::is_constructible_v<SV, const WithStringViewConversionOperator&>);
    static_assert(cuda::std::is_constructible_v<SV, WithStringViewConversionOperator&&>);
  }
}

__host__ __device__ constexpr bool test()
{
  test_from_range<char>();
#if _CCCL_HAS_CHAR8_T()
  test_from_range<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_from_range<char16_t>();
  test_from_range<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_from_range<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

#if _CCCL_HAS_EXCEPTIONS()
void test_exceptions()
{
  // 1. test construction from a type with a throwing data() member
  {
    struct ThrowingData
    {
      char* begin() const
      {
        return nullptr;
      }
      char* end() const
      {
        return nullptr;
      }
      char* data() const
      {
        throw 42;
        return nullptr;
      }
    };
    try
    {
      ThrowingData x;
      (void) cuda::std::string_view(x);
      assert(false);
    }
    catch (int i)
    {
      assert(i == 42);
    }
    catch (...)
    {
      assert(false);
    }
  }

  // 2. test construction from a type with a throwing size() member
  {
    struct ThrowingSize
    {
      char* begin() const
      {
        return nullptr;
      }
      char* end() const
      {
        return nullptr;
      }
      cuda::std::size_t size() const
      {
        throw 42;
        return 0;
      }
    };
    try
    {
      ThrowingSize x;
      (void) cuda::std::string_view(x);
      assert(false);
    }
    catch (int i)
    {
      assert(i == 42);
    }
    catch (...)
    {
      assert(false);
    }
  }
}
#endif // _CCCL_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test());
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // _CCCL_HAS_EXCEPTIONS()
  return 0;
}
