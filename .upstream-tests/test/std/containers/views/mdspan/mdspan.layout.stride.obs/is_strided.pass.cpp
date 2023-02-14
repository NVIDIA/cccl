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

int main(int, char**)
{
    {
        using dexts = cuda::std::dextents<int,2>;
        cuda::std::array<int,2> a{1, 128};

        cuda::std::layout_stride::mapping<dexts> m{dexts{16, 32}, a};

        static_assert( m.is_always_strided() == true, "" );
        assert       ( m.is_strided       () == true );
    }

    return 0;
}
