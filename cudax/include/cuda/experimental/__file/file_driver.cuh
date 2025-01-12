//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__FILE_FILE_DRIVER
#define _CUDAX__FILE_FILE_DRIVER

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/type_identity.h>

#include <cuda/experimental/__file/cufile_api.cuh>

namespace cuda::experimental
{

class file_driver_t
{
public:
  _CCCL_NODISCARD static constexpr file_driver_t __make_file_driver()
  {
    return file_driver_t{};
  }

  struct property
  {
    // Prop obj 1
    // Prop obj 2
    // ...
  };

  template <class Prop>
  auto get(const Prop& prop)
  {
    return prop.get();
  }

  template <class Prop>
  void set(const Prop& prop, ::cuda::std::type_identity_t<typename Prop::value_type> value)
  {
    prop.set(value);
  }

private:
  file_driver_t() = default;
};

inline constexpr file_driver_t file_driver{file_driver_t::__make_file_driver()};

} // namespace cuda::experimental

#endif // _CUDAX__FILE_FILE_DRIVER
