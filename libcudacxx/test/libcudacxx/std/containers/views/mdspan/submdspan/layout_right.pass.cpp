//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// constexpr mdspan& operator=(const mdspan& rhs) = default;

#include <cuda/std/cassert>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "helper.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  constexpr char data[] = {'H', 'O', 'P', 'P', 'E', 'R'};

  { // 1d mdspan
    // ['H', 'O', 'P', 'P', 'E', 'R']
    cuda::std::mdspan md{data, cuda::std::layout_right::mapping{cuda::std::dims<1>{6}}};
    static_assert(md.rank() == 1);
    static_assert(md.rank_dynamic() == 1);
    assert(equal_to(md, "HOPPER"));

    using mdspan_t = decltype(md);
    static_assert(cuda::std::is_same_v<typename mdspan_t::layout_type, cuda::std::layout_right>);

    { // full_extent
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [ x    x    x    x    x    x ]
      cuda::std::mdspan sub = cuda::std::submdspan(md, cuda::std::full_extent);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_right>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 6);
      assert(sub.size() == 6);
      assert(equal_to(sub, "HOPPER"));
    }

    { // Slice of elements from start 0:4
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [ x    x    x    x           ]
      const auto slice      = cuda::std::pair{0, 4};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_right>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 4);
      assert(sub.size() == 4);
      assert(equal_to(sub, "HOPP"));
    }

    { // Slice of elements in the middle 2:5
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [           x    x    x      ]
      const auto slice      = cuda::std::pair{2, 5};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_right>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 3);
      assert(sub.size() == 3);
      assert(equal_to(sub, "PPE"));
    }

    { // Slice of elements in the end 3:6
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [                x    x    x ]
      const auto slice      = cuda::std::pair{3, 6};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_right>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 3);
      assert(sub.size() == 3);
      assert(equal_to(sub, "PER"));
    }

    { // Slice of elements with strided slice without offset, full size and stride 1
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [ x    x    x    x    x    x ] offset
      // [ x    x    x    x    x    x ] size
      // [ x    x    x    x    x    x ] stride
      // [ x    x    x    x    x    x ]
      const cuda::std::strided_slice slice{0, md.extent(0), 1};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 6);
      assert(sub.size() == 6);
      assert(equal_to(sub, "HOPPER"));
    }

    { // Slice of elements with strided slice with offset, full remaining size and stride 1
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [           x    x    x    x ] offset
      // [           x    x    x    x ] size
      // [           x    x    x    x ] stride
      // [           x    x    x    x ]
      const cuda::std::strided_slice slice{2, md.extent(0) - 2, 1};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 4);
      assert(sub.size() == 4);
      assert(equal_to(sub, "PPER"));
    }

    { // Slice of elements with strided slice with offset, smaller size and stride 1
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [           x    x    x    x ] offset
      // [           x    x           ] size
      // [           x    x           ] stride
      // [           x    x           ]
      const cuda::std::strided_slice slice{2, md.extent(0) - 4, 1};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 2);
      assert(sub.size() == 2);
      assert(equal_to(sub, "PP"));
    }

    { // Slice of elements with strided slice without offset, full size and stride 3
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [ x    x    x    x    x    x ] offset
      // [ x    x    x    x    x    x ] size
      // [ x              x           ] stride
      // [ x              x           ]
      const cuda::std::strided_slice slice{0, md.extent(0), 3};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == 3);
      assert(sub.extent(0) == 2);
      assert(sub.size() == 2);
      assert(equal_to(sub, "HP"));
    }

    { // Slice of elements with strided slice with offset, full size and stride 3
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [      x    x    x    x    x ] offset
      // [      x    x    x    x    x ] size
      // [      x              x      ] stride
      // [      x              x      ]
      const cuda::std::strided_slice slice{1, md.extent(0) - 1, 3};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == 3);
      assert(sub.extent(0) == 2);
      assert(sub.size() == 2);
      assert(equal_to(sub, "OE"));
    }

    { // Slice of elements with strided slice with offset, size less equal than stride
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [      x    x    x    x    x ] offset
      // [      x    x    x           ] size
      // [      x                     ] stride
      // [      x                     ]
      const cuda::std::strided_slice slice{1, 3, 3};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 1);
      static_assert(sub.rank_dynamic() == 1);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == 1);
      assert(sub.extent(0) == 1);
      assert(sub.size() == 1);
      assert(equal_to(sub, "O"));
    }

    { // Single element, with integral constant
      // ['H', 'O', 'P', 'P', 'E', 'R']
      // [                x           ]
      const auto slice      = cuda::std::integral_constant<size_t, 3>{};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice);

      static_assert(sub.rank() == 0);
      static_assert(sub.rank_dynamic() == 0);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_right>);

      assert(sub.size() == 1);
      assert(equal_to(sub, "P"));
    }
  }

  { // 2d mdspan
    // ['H', 'O', 'P']
    // ['P', 'E', 'R']
    cuda::std::mdspan md{data, cuda::std::layout_right::mapping{cuda::std::dims<2>{2, 3}}};
    static_assert(md.rank() == 2);
    static_assert(md.rank_dynamic() == 2);

    assert(md.stride(0) == md.extent(1));
    assert(md.stride(1) == 1);
    assert(md.extent(0) == 2);
    assert(md.extent(1) == 3);
    assert(md.size() == 6);
    assert(equal_to(md, {"HOP", "PER"}));

    { // full_extent
      // ['H', 'O', 'P'] [ x ] [ x    x    x ]
      // ['P', 'E', 'R'] [ x ] [ x    x    x ]
      cuda::std::mdspan sub = cuda::std::submdspan(md, cuda::std::full_extent, cuda::std::full_extent);

      static_assert(sub.rank() == 2);
      static_assert(sub.rank_dynamic() == 2);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_right>);

      assert(sub.stride(0) == md.extent(1));
      assert(sub.stride(1) == 1);
      assert(sub.extent(0) == md.extent(0));
      assert(sub.extent(1) == md.extent(1));
      assert(sub.size() == md.size());
      assert(equal_to(sub, {"HOP", "PER"}));
    }

    { // full extent, then slice of elements from start 0:1
      // ['H', 'O', 'P'] [ x ] [ x           ]
      // ['P', 'E', 'R'] [ x ] [ x           ]
      const auto slice2     = cuda::std::pair{0, 1};
      cuda::std::mdspan sub = cuda::std::submdspan(md, cuda::std::full_extent, slice2);

      static_assert(sub.rank() == 2);
      static_assert(sub.rank_dynamic() == 2);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_right>);

      assert(sub.stride(0) == 1);
      assert(sub.stride(1) == md.stride(1));
      assert(sub.extent(0) == md.extent(0));
      assert(sub.extent(1) == 1);
      assert(sub.size() == 2);
      assert(equal_to(sub, {"H", "O"}));
    }

    { // Slice of elements from start 1:2, then full extent
      // ['H', 'O', 'P'] [   ] [             ]
      // ['P', 'E', 'R'] [ x ] [ x    x    x ]
      const auto slice1     = cuda::std::pair{1, 2};
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
      assert(equal_to(sub, {"PER", ""}));
    }

    { // Slice of elements from middle 1:2, then strided_slice without offset, full size and stride 1
      // ['H', 'O', 'P'] [   ] [             ]
      // ['P', 'E', 'R'] [ x ] [ x    x    x ]
      const auto slice1 = cuda::std::pair{1, 2};
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
      assert(equal_to(sub, {"PER", ""}));
    }

    { // Slice of elements from middle 1:2, then strided_slice with offset, full size and stride 1
      // ['H', 'O', 'P'] [   ] [             ]
      // ['P', 'E', 'R'] [ x ] [      x    x ]
      const auto slice1 = cuda::std::pair{1, 2};
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
      assert(equal_to(sub, {"ER", ""}));
    }

    { // Slice of elements from middle 1:2, then strided_slice without offset, full size and stride 2
      // ['H', 'O', 'P'] [   ] [             ]
      // ['P', 'E', 'R'] [ x ] [ x           ]
      const auto slice1 = cuda::std::pair{1, 2};
      const cuda::std::strided_slice slice2{0, 2, 2};
      cuda::std::mdspan sub = cuda::std::submdspan(md, slice1, slice2);

      static_assert(sub.rank() == 2);
      static_assert(sub.rank_dynamic() == 2);

      using submdspan_t = decltype(sub);
      static_assert(cuda::std::is_same_v<typename submdspan_t::layout_type, cuda::std::layout_stride>);

      assert(sub.stride(0) == md.stride(0));
      assert(sub.stride(1) == 1);
      assert(sub.extent(0) == 1);
      assert(sub.extent(1) == 1);
      assert(sub.size() == 1);
      assert(equal_to(sub, {"P", ""}));
    }

    { // Slice of elements from middle 1:2, then index
      // ['H', 'O', 'P'] [   ] [             ]
      // ['P', 'E', 'R'] [ x ] [           x ]
      const auto slice1     = cuda::std::pair{1, 2};
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

  return true;
}

int main(int, char**)
{
  test();
  // static_assert(test(), "");
  return 0;
}
