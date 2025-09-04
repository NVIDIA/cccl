//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Type-erased resources associated to contexts

#pragma once

#include <cuda/__cccl_config>

#include <memory>
#include <vector>

#include <cuda/experimental/__stf/utility/core.cuh>

#include <cuda_runtime.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

namespace cuda::experimental::stf
{

//! Generic container for a resource that needs to be retained until a context has consumed them
class ctx_resource {
public:
    //! Release asynchronously
    virtual void release(cudaStream_t) = 0;
};

class ctx_resource_set {
public:
    //! Store a resource until it is released
    void add(::std::shared_ptr<ctx_resource> r) {
        resources.push_back(mv(r));
    }

    //! Release all resources asynchronously
    void release(cudaStream_t stream) {
        for (auto &r: resources) {
            r->release(stream);
        }
        resources.clear();
    }

private:
    ::std::vector<::std::shared_ptr<ctx_resource>> resources;
};

} // end namespace cuda::experimental::stf
