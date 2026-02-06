//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>

// The public namespace.
namespace ns
{
// The inline ABI namespace.
inline namespace v1
{
// Forward declarations.
struct unrev_struct;
_CCCL_DECLARE_ABI_REV(struct my_struct, 2);
_CCCL_DECLARE_ABI_REV(class my_class, 1);
_CCCL_DECLARE_ABI_REV(template(class T) class my_template_class, 3);

// Definitions.
struct unrev_struct
{
  int value;
};

struct _CCCL_REV(my_struct)
{
  int value;
  __host__ __device__ int method();
};

__host__ __device__ int my_struct::method()
{
  return value;
}

class _CCCL_REV(my_class)
{
public:
  __host__ __device__ int my_struct_interact(const my_struct& v)
  {
    return v.value;
  }
};

template <class T>
class _CCCL_REV(my_template_class)
{};

template <>
class _CCCL_REV(my_template_class)<my_struct> : public my_struct
{};

template <class T>
__host__ __device__ auto adl_fn(const T& v)
{
  return v.value;
}
} // namespace v1
} // namespace ns

__host__ __device__ void test()
{
  // 1. Test that the symbols are in the expected namespace.
  static_assert(cuda::std::is_same_v<ns::my_class, ns::v1::__r1::my_class>);
  static_assert(cuda::std::is_same_v<ns::v1::__r1::my_class, ns::v1::__my_class_rev_ns::my_class>);

  static_assert(cuda::std::is_same_v<ns::my_struct, ns::v1::__r2::my_struct>);
  static_assert(cuda::std::is_same_v<ns::v1::__r2::my_struct, ns::v1::__my_struct_rev_ns::my_struct>);

  static_assert(cuda::std::is_same_v<ns::my_template_class<int>, ns::v1::__r3::my_template_class<int>>);
  static_assert(cuda::std::is_same_v<ns::v1::__r3::my_template_class<int>,
                                     ns::v1::__my_template_class_rev_ns::my_template_class<int>>);

  // 2. Test construction.
  [[maybe_unused]] ns::unrev_struct us{};
  [[maybe_unused]] ns::my_class c{};
  [[maybe_unused]] ns::my_struct s{};
  [[maybe_unused]] ns::my_template_class<int> tci{};
  [[maybe_unused]] ns::my_template_class<ns::my_struct> tcs{};

  // 3. Test ADL.
  [[maybe_unused]] const auto result1 = adl_fn(us);
  [[maybe_unused]] const auto result2 = adl_fn(s);
  [[maybe_unused]] const auto result3 = adl_fn(tcs);

  // 4. Test interaction.
  [[maybe_unused]] const auto result4 = c.my_struct_interact(s);
  [[maybe_unused]] const auto result5 = c.my_struct_interact(tcs);
}

int main(int, char**)
{
  return 0;
}
