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
    cuda::std::array<int,1> storage{1};

    {
        cuda::std::mdspan<int, cuda::std::dextents<int,1>> m;

        assert( m.empty() == true );
    }

    {
        cuda::std::mdspan<int, cuda::std::dextents<int,1>> m{ storage.data(), 0 };

        assert( m.empty() == true );
    }

    {
        cuda::std::mdspan<int, cuda::std::dextents<int,1>> m{ storage.data(), 2 };

        assert( m.empty() == false );
    }

    {
        cuda::std::mdspan<int, cuda::std::dextents<int,2>> m{ storage.data(), 2, 0 };

        assert( m.empty() == true );
    }

    {
        cuda::std::mdspan<int, cuda::std::dextents<int,2>> m{ storage.data(), 2, 2 };

        assert( m.empty() == false );
    }

    return 0;
}
