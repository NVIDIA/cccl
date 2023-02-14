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
        typedef int    data_t ;
        typedef size_t index_t;

        cuda::std::mdspan<data_t, cuda::std::dextents<index_t,1>> m;

        static_assert(m.is_exhaustive() == true, "");

        assert(m.data_handle()    == nullptr);
        assert(m.rank()           == 1      );
        assert(m.rank_dynamic()   == 1      );
        assert(m.extent(0)        == 0      );
        assert(m.static_extent(0) == dyn    );
        assert(m.stride(0)        == 1      );
        assert(m.size()           == 0      );
        assert(m.empty()          == true   );
    }

    return 0;
}
