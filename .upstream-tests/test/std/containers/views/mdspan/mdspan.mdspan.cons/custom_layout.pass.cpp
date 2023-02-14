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
    using map_t = Foo::layout_foo::template mapping<cuda::std::dextents<size_t ,2>>;

    {
        using  data_t = int;
        using   lay_t = Foo::layout_foo;
        using index_t = size_t;

        cuda::std::array<data_t, 1> d{42};
        lay_t::mapping<cuda::std::extents<index_t,dyn,dyn>> map{cuda::std::dextents<index_t,2>{64, 128}};
        cuda::std::mdspan<data_t, cuda::std::extents<index_t,dyn,dyn>, lay_t> m{ d.data(), map };

        CHECK_MDSPAN_EXTENT(m,d,64,128);
    }

    return 0;
}
