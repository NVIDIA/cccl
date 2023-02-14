//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

//UNSUPPORTED: c++11, nvrtc && nvcc-12.0, nvrtc && nvcc-12.1

#include <cuda/std/mdspan>
#include <cuda/std/cassert>

constexpr auto dyn = cuda::std::dynamic_extent;

int main(int, char**)
{
    using index_t = int;

    {
        cuda::std::extents<index_t,16> e;
        cuda::std::array<index_t,1> a{1};
        cuda::std::layout_stride::mapping<cuda::std::extents<index_t,16>> m{e, a};

        static_assert( m.is_always_unique() == true, "" );
        assert       ( m.is_unique       () == true );
    }

    {
        cuda::std::extents<index_t,dyn,dyn> e{16, 32};
        cuda::std::array<index_t,2> a{1,16};
        cuda::std::layout_stride::mapping<cuda::std::extents<index_t,dyn,dyn>> m{e, a};

        static_assert( m.is_always_unique() == true, "" );
        assert       ( m.is_unique       () == true );
    }

    return 0;
}
