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
    {
        cuda::std::array<int, 1> d{42};
        cuda::std::extents<int,dyn,dyn> e{64, 128};
        cuda::std::mdspan<int, cuda::std::extents<int,dyn,dyn>> m{ d.data(), e };

        assert( &m.extents() == &m.mapping().extents() );
    }

    return 0;
}

