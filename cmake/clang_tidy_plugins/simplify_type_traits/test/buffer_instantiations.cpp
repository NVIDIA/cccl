// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-simplify-type-traits
// CLANG_TIDY_CONFIG_END

#include <cuda/std/type_traits>

// These declarations isolate the rewrite from CUDA runtime dependencies.
// CHECK-MESSAGES-NOT: warning:
namespace cuda
{
inline namespace abi
{
namespace mr
{
struct device_accessible {};
struct host_accessible {};
}
template <typename T, typename... Properties>
struct buffer
{
  buffer() = default;
  explicit buffer(int) {}

  template <typename... OtherProperties>
  buffer(buffer<T, OtherProperties...>&&) {}
};
template <typename T>
using device_buffer = buffer<T, mr::device_accessible>;
template <typename T>
using host_buffer = buffer<T, mr::host_accessible>;
}
}

// This control must still diagnose an explicitly written accessibility property.
using explicit_device = cuda::buffer<int, cuda::mr::device_accessible>;
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use simplified type alias [cccl-simplify-type-traits]
// CHECK-MESSAGES-NOT: warning:
// CHECK-FIXES-LABEL: - DiagnosticName: cccl-simplify-type-traits
// CHECK-FIXES: ReplacementText: {{'?}}::cuda::device_buffer{{'?}}{{$}}
// CHECK-FIXES: ReplacementText: ''
// CHECK-FIXES-NOT: DiagnosticName:

struct custom_property {};

// A specialization's properties must not replace the original dependent arguments.
template <typename Property>
struct single_property
{
  using type = cuda::buffer<int, Property>;
  type value;
};
template <typename... Properties>
struct property_pack
{
  using type = cuda::buffer<int, Properties...>;
  type value;
};
template <typename FirstProperty, typename... RestProperties>
auto make_buffer() -> cuda::buffer<int, FirstProperty, RestProperties...>
{
  cuda::buffer<int, FirstProperty, RestProperties...> result;
  return result;
}
template <typename FirstProperty, typename... RestProperties>
auto make_buffer_expression()
{
  return cuda::buffer<int, FirstProperty, RestProperties...>{};
}

// Inherited constructors must not create diagnostics at the using declaration.
template <typename T, typename... Properties>
struct resizable_buffer : cuda::buffer<T, Properties...>
{
  using base_type = cuda::buffer<T, Properties...>;
  using base_type::base_type;
};

template <typename T>
struct matcher {};
template <typename T, typename... Properties>
auto equals(const cuda::buffer<T, Properties...>&)
  -> matcher<cuda::buffer<T, Properties...>>
{
  return {};
}

// Substitution of the template name must not make that name eligible for replacement.
template <template <typename, typename...> class Container, typename Property>
struct container_template
{
  using type = Container<int, Property>;
  type value;
};

void instantiate_buffers()
{
  [[maybe_unused]] single_property<cuda::mr::host_accessible> single_host;
  [[maybe_unused]] single_property<cuda::mr::device_accessible> single_device;
  [[maybe_unused]] single_property<custom_property> single_custom;
  [[maybe_unused]] property_pack<> empty;
  [[maybe_unused]] property_pack<cuda::mr::host_accessible> host;
  [[maybe_unused]] property_pack<cuda::mr::device_accessible> device;
  [[maybe_unused]] property_pack<cuda::mr::host_accessible, cuda::mr::device_accessible> both;
  [[maybe_unused]] property_pack<cuda::mr::device_accessible, custom_property> extra;

  [[maybe_unused]] auto made_host = make_buffer<cuda::mr::host_accessible>();
  [[maybe_unused]] auto made_device = make_buffer<cuda::mr::device_accessible>();
  [[maybe_unused]] auto made_extra = make_buffer<cuda::mr::device_accessible, custom_property>();
  [[maybe_unused]] auto expression_host = make_buffer_expression<cuda::mr::host_accessible>();
  [[maybe_unused]] auto expression_device = make_buffer_expression<cuda::mr::device_accessible>();
  [[maybe_unused]] auto expression_extra = make_buffer_expression<cuda::mr::device_accessible, custom_property>();

  [[maybe_unused]] resizable_buffer<int, cuda::mr::host_accessible> resizable_host{1};
  [[maybe_unused]] resizable_buffer<int, cuda::mr::device_accessible> resizable_device{1};
  [[maybe_unused]] resizable_buffer<int, cuda::mr::device_accessible, custom_property> resizable_extra{1};
  [[maybe_unused]] cuda::device_buffer<int> converted_host{cuda::host_buffer<int>{}};
  [[maybe_unused]] cuda::host_buffer<int> converted_device{cuda::device_buffer<int>{}};

  [[maybe_unused]] auto host_matcher = equals(host.value);
  [[maybe_unused]] auto device_matcher = equals(device.value);
  [[maybe_unused]] auto extra_matcher = equals(extra.value);
  [[maybe_unused]] container_template<cuda::buffer, cuda::mr::host_accessible> template_host;
  [[maybe_unused]] container_template<cuda::buffer, cuda::mr::device_accessible> template_device;
  [[maybe_unused]] container_template<cuda::buffer, custom_property> template_custom;

  static_assert(cuda::std::is_same_v<decltype(made_host), cuda::host_buffer<int>>);
  static_assert(cuda::std::is_same_v<decltype(made_device), cuda::device_buffer<int>>);
  static_assert(!cuda::std::is_same_v<decltype(made_extra), cuda::device_buffer<int>>);
  static_assert(cuda::std::is_same_v<decltype(host_matcher), matcher<cuda::host_buffer<int>>>);
  static_assert(cuda::std::is_same_v<decltype(device_matcher), matcher<cuda::device_buffer<int>>>);
}
// clang-format on
