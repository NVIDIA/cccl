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
    using ext2d_t = cuda::std::extents<index_t,dyn,dyn>;

    {
        ext2d_t e{16, 32};
        cuda::std::array<index_t,2> a{1,16};
        cuda::std::layout_stride::mapping<ext2d_t> m{e, a};

        assert( m.extents() == e );
    }

    return 0;
}

