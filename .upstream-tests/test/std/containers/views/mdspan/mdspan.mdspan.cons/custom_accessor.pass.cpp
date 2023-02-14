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
#include "../mdspan.mdspan.util/mdspan_util.hpp"
#include "../foo_customizations.hpp"

constexpr auto dyn = cuda::std::dynamic_extent;

int main(int, char**)
{
    {
        using  data_t = int;
        using   acc_t = Foo::foo_accessor<data_t>;
        using index_t = size_t;

        cuda::std::array<data_t, 1> d{42};
        cuda::std::layout_left::mapping<cuda::std::extents<index_t,dyn,dyn>> map{cuda::std::dextents<index_t,2>{64, 128}};
        acc_t a;
        cuda::std::mdspan<data_t, cuda::std::extents<index_t,dyn,dyn>, cuda::std::layout_left, acc_t> m{ d.data(), map, a };

        static_assert(m.is_exhaustive(), "");
        //assert(m.data_handle()  == d.data());
        assert(m.rank()         == 2       );
        assert(m.rank_dynamic() == 2       );
        assert(m.extent(0)      == 64      );
        assert(m.extent(1)      == 128     );
    }

    return 0;
}
