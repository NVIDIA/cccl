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
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"
#include "test_iterators.h"
#include "test_range.h"

template <class CharT>
struct TestBase
{
  __host__ __device__ constexpr const CharT* range_data() const
  {
    return range_data_;
  }
  __host__ __device__ constexpr cuda::std::size_t range_size() const
  {
    return cuda::std::char_traits<CharT>::length(range_data());
  }
  __host__ __device__ constexpr const CharT* conv_data() const
  {
    return conv_data_;
  }

  __host__ __device__ constexpr const CharT* begin() const
  {
    return range_data();
  }
  __host__ __device__ constexpr const CharT* end() const
  {
    return range_data() + range_size();
  }

  const CharT* range_data_;
  const CharT* conv_data_;
};

template <class CharT>
__host__ __device__ constexpr void test_from_range()
{
  using SV = cuda::std::basic_string_view<CharT>;
  using TB = TestBase<CharT>;

  const auto range_data = TEST_STRLIT(CharT, "range_data");
  const auto conv_data  = TEST_STRLIT(CharT, "conv_data");

  // 1. test construction from cuda::std::array
  {
    cuda::std::array<CharT, 3> arr{TEST_CHARLIT(CharT, 'f'), TEST_CHARLIT(CharT, 'o'), TEST_CHARLIT(CharT, 'o')};
    auto sv = SV(arr);

    static_assert(cuda::std::is_same_v<decltype(sv), SV>);
    assert(sv.size() == arr.size());
    assert(sv.data() == arr.data());
  }

  // 2. test construction from a type with a non-const conversion operator
  {
    struct NonConstConversionOperator : TB
    {
      __host__ __device__ constexpr operator SV()
      {
        return TB::conv_data();
      }
    };

    static_assert(cuda::std::is_constructible_v<SV, NonConstConversionOperator>);
    static_assert(!cuda::std::is_convertible_v<const NonConstConversionOperator&, SV>);

    NonConstConversionOperator nc{range_data, conv_data};
    SV sv = nc;
    assert(sv == nc.conv_data());
  }

  // 3. test construction from a type with a const conversion operator
  {
    struct ConstConversionOperator : TB
    {
      __host__ __device__ constexpr operator SV() const
      {
        return TB::conv_data();
      }
    };

    static_assert(cuda::std::is_constructible_v<SV, ConstConversionOperator>);
    static_assert(cuda::std::is_constructible_v<SV, const ConstConversionOperator&>);

    {
      ConstConversionOperator cv{range_data, conv_data};
      SV sv = cv;
      assert(sv == cv.conv_data());
    }
    {
      const ConstConversionOperator cv{range_data, conv_data};
      SV sv = cv;
      assert(sv == cv.conv_data());
    }
  }

  // 4. test construction from a type with a deleted conversion operator
  // disabled for nvc++ with char16_t and char32_t, see nvbug 5277567
#if _CCCL_COMPILER(NVHPC)
  if constexpr (!cuda::std::is_same_v<CharT, char16_t> && !cuda::std::is_same_v<CharT, char32_t>)
#endif // _CCCL_COMPILER(NVHPC)
  {
    struct DeletedConversionOperator : TB
    {
      operator SV() = delete;
    };

    static_assert(cuda::std::is_constructible_v<SV, DeletedConversionOperator>);
    static_assert(cuda::std::is_constructible_v<SV, const DeletedConversionOperator>);

    {
      DeletedConversionOperator cv{range_data, conv_data};
      SV sv = SV(cv);
      assert(sv == cv.range_data());
    }
    {
      const DeletedConversionOperator cv{range_data, conv_data};
      SV sv = SV(cv);
      assert(sv == cv.range_data());
    }
  }

  // 5. test construction from a type with a deleted const conversion operator
  // disabled for nvc++ with char16_t and char32_t, see nvbug 5277567
#if _CCCL_COMPILER(NVHPC)
  if constexpr (!cuda::std::is_same_v<CharT, char16_t> && !cuda::std::is_same_v<CharT, char32_t>)
#endif // _CCCL_COMPILER(NVHPC)
  {
    struct DeletedConstConversionOperator : TB
    {
      operator SV() const = delete;
    };

    static_assert(cuda::std::is_constructible_v<SV, DeletedConstConversionOperator>);
    static_assert(cuda::std::is_constructible_v<SV, const DeletedConstConversionOperator>);

    {
      DeletedConstConversionOperator cv{range_data, conv_data};
      SV sv = SV(cv);
      assert(sv == cv.range_data());
    }
    {
      const DeletedConstConversionOperator cv{range_data, conv_data};
      SV sv = SV(cv);
      assert(sv == cv.range_data());
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
      __host__ __device__ constexpr const CharT* begin() const
      {
        return nullptr;
      }
      __host__ __device__ constexpr const CharT* end() const
      {
        return nullptr;
      }
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
