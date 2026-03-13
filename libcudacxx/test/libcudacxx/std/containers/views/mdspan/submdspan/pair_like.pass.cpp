//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/mdspan>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "helper.h"
#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <array>
#  include <tuple>
#  include <utility>
#endif // !TEST_COMPILER(NVRTC)

template <class Tuple>
__host__ __device__ constexpr void test()
{
  constexpr char data[] = {'H', 'O', 'P', 'P', 'E', 'R'};

  { // 1d mdspan
    // ['H', 'O', 'P', 'P', 'E', 'R']
    cuda::std::dims<1> extent{6};
    cuda::std::layout_left::mapping<cuda::std::dims<1>> mapping{extent};
    cuda::std::mdspan md{data, mapping};
    static_assert(md.rank() == 1);
    static_assert(md.rank_dynamic() == 1);
    assert(equal_to(md, "HOPPER"));

    using mdspan_t = decltype(md);
    static_assert(cuda::std::is_same_v<typename mdspan_t::layout_type, cuda::std::layout_left>);

    { // Slice of elements from start 0:4
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [ x    x    x    x           ]
      const auto slice      = Tuple{0, 4};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_left>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 4);
      assert(sub.size() == 4);
      assert(equal_to(sub, "HOPP"));
    }

    { // Slice of elements in the middle 2:5
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [           x    x    x      ]
      const auto slice      = Tuple{2, 5};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_left>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 3);
      assert(sub.size() == 3);
      assert(equal_to(sub, "PPE"));
    }

    { // Slice of elements in the end 3:6
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [                x    x    x ]
      const auto slice      = Tuple{3, 6};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_left>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 3);
      assert(sub.size() == 3);
      assert(equal_to(sub, "PER"));
    }
  }

  { // 2d mdspan
    // ['H', 'P', 'E']
    // ['O', 'P', 'R']
    cuda::std::dims<2> extent{2, 3};
    cuda::std::layout_left::mapping<cuda::std::dims<2>> mapping{extent};
    cuda::std::mdspan md{data, mapping};
    static_assert(md.rank() == 2);
    static_assert(md.rank_dynamic() == 2);

    assert(md.stride(0) == 1);
    assert(md.stride(1) == md.extent(0));
    assert(md.extent(0) == 2);
    assert(md.extent(1) == 3);
    assert(md.size() == 6);
    assert(equal_to(md, {"HPE", "OPR"}));

    { // full extent, then slice of elements from start 0:2
      // ['H', 'P', 'E'] [ x ] [ x    x      ]
      // ['O', 'P', 'R'] [ x ] [ x    x      ]
      const auto slice2     = Tuple{0, 2};
      cuda::std::mdspan sub = cuda::std::submdspan(md, cuda::std::full_extent, slice2);

      static_assert(sub.rank() == 2);
      static_assert(sub.rank_dynamic() == 2);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_left>);

      assert(sub.stride(0) == md.stride(0));
      assert(sub.stride(1) == md.stride(1));
      assert(sub.extent(0) == md.extent(0));
      assert(sub.extent(1) == 2);
      assert(sub.size() == 4);
      assert(equal_to(sub, {"HP", "OP"}));
    }

    { // Slice of elements from start 0:1, then full extent
      // ['H', 'P', 'E'] [ x ] [ x    x    x ]
      // ['O', 'P', 'R'] [   ] [             ]
      const auto slice1     = Tuple{0, 1};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice1, cuda::std::full_extent);

      static_assert(sub.rank() == 2);
      static_assert(sub.rank_dynamic() == 2);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == md.stride(0));
      assert(sub.stride(1) == md.stride(1));
      assert(sub.extent(0) == 1);
      assert(sub.extent(1) == md.extent(1));
      assert(sub.size() == 3);
      assert(equal_to(sub, {"HPE", ""}));
    }

    { // Slice of elements from middle 1:2, then strided_slice without offset, full size and stride 1
      // ['H', 'P', 'E'] [   ] [             ]
      // ['O', 'P', 'R'] [ x ] [ x    x    x ]
      const auto slice1 = Tuple{1, 2};
      const cuda::std::strided_slice slice2{0, md.extent(1), 1};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice1, slice2);

      static_assert(sub.rank() == 2);
      static_assert(sub.rank_dynamic() == 2);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == md.stride(0));
      assert(sub.stride(1) == md.stride(1));
      assert(sub.extent(0) == 1);
      assert(sub.extent(1) == md.extent(1));
      assert(sub.size() == 3);
      assert(equal_to(sub, {"OPR", ""}));
    }

    { // Slice of elements from middle 1:2, then strided_slice with offset, full size and stride 1
      // ['H', 'P', 'E'] [   ] [             ]
      // ['O', 'P', 'R'] [ x ] [      x    x ]
      const auto slice1 = Tuple{1, 2};
      const cuda::std::strided_slice slice2{1, md.extent(1) - 1, 1};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice1, slice2);

      static_assert(sub.rank() == 2);
      static_assert(sub.rank_dynamic() == 2);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == md.stride(0));
      assert(sub.stride(1) == md.stride(1));
      assert(sub.extent(0) == 1);
      assert(sub.extent(1) == md.extent(1) - 1);
      assert(sub.size() == 2);
      assert(equal_to(sub, {"PR", ""}));
    }

    { // Slice of elements from middle 1:2, then strided_slice without offset, full size and stride 2
      // ['H', 'P', 'E'] [   ] [             ]
      // ['O', 'P', 'R'] [ x ] [ x         x ]
      const auto slice1 = Tuple{1, 2};
      const cuda::std::strided_slice slice2{0, md.extent(1), 2};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice1, slice2);

      static_assert(sub.rank() == 2);
      static_assert(sub.rank_dynamic() == 2);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == md.stride(0));
      assert(sub.stride(1) == 2 * md.stride(1));
      assert(sub.extent(0) == 1);
      assert(sub.extent(1) == md.extent(1) - 1);
      assert(sub.size() == 2);
      assert(equal_to(sub, {"OR", ""}));
    }

    { // Slice of elements from middle 1:2, then index
      // ['H', 'P', 'E'] [   ] [             ]
      // ['O', 'P', 'R'] [ x ] [           x ]
      const auto slice1     = Tuple{1, 2};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice1, 2);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == md.stride(0));
      assert(sub.extent(0) == 1);
      assert(sub.size() == 1);
      assert(equal_to(sub, "R"));
    }
  }
}

__host__ __device__ constexpr bool test()
{
  test<cuda::std::array<int, 2>>();
  test<cuda::std::pair<int, int>>();
  test<cuda::std::tuple<int, int>>();
  test<cuda::std::complex<int>>();

#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, ({
                 test<std::array<int, 2>>();
                 test<std::pair<int, int>>();
                 test<std::tuple<int, int>>();
               }))
#endif // !TEST_COMPILER(NVRTC)

  return true;
}

int main(int, char**)
{
  test();
#if !_CCCL_COMPILER(GCC, <, 11) // gcc-10 complains about __submdspan_offset not being constexpr...
  static_assert(test());
#endif // !_CCCL_COMPILER(GCC, <, 11)
  return 0;
}
