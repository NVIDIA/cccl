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
    using index_t = size_t;

    {
        cuda::std::layout_right::mapping<cuda::std::dextents<index_t,1>> m;

        static_assert( m.is_always_exhaustive() == true, "" );
        assert       ( m.is_exhaustive       () == true );
    }


    {
        cuda::std::extents<index_t,dyn,dyn> e{16, 32};
        cuda::std::layout_right::mapping<cuda::std::extents<index_t,dyn,dyn>> m{ e };

        static_assert( m.is_always_exhaustive() == true, "" );
        assert       ( m.is_exhaustive       () == true );
    }

    return 0;
}
