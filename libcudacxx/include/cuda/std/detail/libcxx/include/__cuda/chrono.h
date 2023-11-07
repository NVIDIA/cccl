// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_CHRONO_H
#define _LIBCUDACXX___CUDA_CHRONO_H

#ifndef __cuda_std__
#error "<__cuda/chrono> should only be included in from <cuda/std/chrono>"
#endif // __cuda_std__

#include <nv/target>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace chrono {

inline _LIBCUDACXX_INLINE_VISIBILITY
system_clock::time_point system_clock::now() noexcept
{
NV_DISPATCH_TARGET(
NV_IS_DEVICE, (
    uint64_t __time;
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(__time)::);
    return time_point(duration_cast<duration>(nanoseconds(__time)));
),
NV_IS_HOST, (
    return time_point(duration_cast<duration>(nanoseconds(
            ::std::chrono::duration_cast<::std::chrono::nanoseconds>(
                ::std::chrono::system_clock::now().time_since_epoch()
            ).count()
           )));
));
}

inline _LIBCUDACXX_INLINE_VISIBILITY
time_t system_clock::to_time_t(const system_clock::time_point& __t) noexcept
{
    return time_t(duration_cast<seconds>(__t.time_since_epoch()).count());
}

inline _LIBCUDACXX_INLINE_VISIBILITY
system_clock::time_point system_clock::from_time_t(time_t __t) noexcept
{
    return time_point(seconds(__t));;
}
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CUDA_CHRONO_H
