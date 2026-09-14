//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda.h>

#include "device_copy_codegen.h"
#include <cccl/c/device_copy.h>
#include <hostjit/codegen/cub_call.hpp>
#include <hostjit/jit_compiler.hpp>
#include <util/build_utils.h>

namespace
{
constexpr const char* device_copy_fn_name = "cccl_jit_device_copy";

using device_copy_fn_t = int (*)(
  const void* source_data,
  unsigned long long source_byte_offset,
  const int64_t* source_shape,
  const int64_t* source_strides,
  void* destination_data,
  unsigned long long destination_byte_offset,
  const int64_t* destination_shape,
  const int64_t* destination_strides,
  void* stream);

bool is_power_of_two(size_t value)
{
  return value != 0 && (value & (value - 1)) == 0;
}

bool effective_address_is_aligned(const void* data, uint64_t byte_offset, size_t alignment)
{
  if (data == nullptr || alignment == 0)
  {
    return false;
  }
  if (alignment == 1)
  {
    return true;
  }

  const auto base = reinterpret_cast<std::uintptr_t>(data);
  if (byte_offset > static_cast<uint64_t>(std::numeric_limits<std::uintptr_t>::max() - base))
  {
    return false;
  }

  const auto effective_address = base + static_cast<std::uintptr_t>(byte_offset);
  return (effective_address % alignment) == 0;
}

bool is_contiguous_layout(cccl_device_copy_layout_kind_t layout)
{
  return layout == CCCL_DEVICE_COPY_LAYOUT_RIGHT || layout == CCCL_DEVICE_COPY_LAYOUT_LEFT;
}

bool is_layout_stride_relaxed(cccl_device_copy_layout_kind_t layout)
{
  return layout == CCCL_DEVICE_COPY_LAYOUT_STRIDE_RELAXED;
}

bool is_layout_stride(cccl_device_copy_layout_kind_t layout)
{
  return layout == CCCL_DEVICE_COPY_LAYOUT_STRIDE;
}

bool is_strided_layout(cccl_device_copy_layout_kind_t layout)
{
  return is_layout_stride(layout) || is_layout_stride_relaxed(layout);
}

bool is_supported_layout(cccl_device_copy_layout_kind_t layout)
{
  return is_contiguous_layout(layout) || is_strided_layout(layout);
}

bool all_runtime_metadata(const cccl_device_copy_axis_metadata_t* metadata, size_t rank)
{
  if (metadata == nullptr)
  {
    return false;
  }

  for (size_t axis = 0; axis < rank; ++axis)
  {
    if (metadata[axis].kind != CCCL_DEVICE_COPY_AXIS_RUNTIME || metadata[axis].value != 0)
    {
      return false;
    }
  }

  return true;
}

bool valid_shape_metadata(const cccl_device_copy_axis_metadata_t* metadata, size_t rank)
{
  if (metadata == nullptr)
  {
    return false;
  }

  for (size_t axis = 0; axis < rank; ++axis)
  {
    switch (metadata[axis].kind)
    {
      case CCCL_DEVICE_COPY_AXIS_RUNTIME:
        if (metadata[axis].value != 0)
        {
          return false;
        }
        break;
      case CCCL_DEVICE_COPY_AXIS_STATIC:
        if (metadata[axis].value < 0)
        {
          return false;
        }
        break;
      default:
        return false;
    }
  }

  return true;
}

bool has_runtime_metadata(const cccl_device_copy_axis_metadata_t* metadata, size_t rank)
{
  for (size_t axis = 0; axis < rank; ++axis)
  {
    if (metadata[axis].kind == CCCL_DEVICE_COPY_AXIS_RUNTIME)
    {
      return true;
    }
  }

  return false;
}

std::unique_ptr<cccl_device_copy_axis_metadata_t[]>
retain_axis_metadata(const cccl_device_copy_axis_metadata_t* metadata, size_t rank)
{
  auto retained = std::make_unique<cccl_device_copy_axis_metadata_t[]>(rank);
  for (size_t axis = 0; axis < rank; ++axis)
  {
    retained[axis] = metadata[axis];
  }

  return retained;
}

std::unique_ptr<cccl_device_copy_axis_metadata_t[]>
retain_stride_metadata(cccl_device_copy_view_build_t view, size_t rank)
{
  if (!is_strided_layout(view.layout))
  {
    return {};
  }

  return retain_axis_metadata(view.strides, rank);
}

std::unique_ptr<char[]> retain_source(const std::string& source)
{
  auto retained = std::make_unique<char[]>(source.size() + 1);
  for (size_t i = 0; i < source.size(); ++i)
  {
    retained[i] = source[i];
  }
  retained[source.size()] = '\0';
  return retained;
}

bool validate_view_build(cccl_device_copy_view_build_t view, size_t rank)
{
  if (!is_supported_layout(view.layout))
  {
    return false;
  }

  if (is_strided_layout(view.layout))
  {
    return all_runtime_metadata(view.strides, rank);
  }

  return true;
}

bool product_is_representable(size_t rank, const int64_t* shape)
{
  uint64_t product = 1;
  for (size_t axis = 0; axis < rank; ++axis)
  {
    if (shape[axis] == 0)
    {
      return true;
    }

    const auto extent = static_cast<uint64_t>(shape[axis]);
    if (product > std::numeric_limits<uint64_t>::max() / extent)
    {
      return false;
    }
    product *= extent;
  }

  return true;
}

bool strided_span_is_representable(size_t rank, const int64_t* shape, const int64_t* strides)
{
  uint64_t negative_offset = 0;
  uint64_t positive_span   = 1;

  for (size_t axis = 0; axis < rank; ++axis)
  {
    if (shape[axis] == 0)
    {
      return true;
    }
    if (shape[axis] == 1 || strides[axis] == 0)
    {
      continue;
    }

    const auto extent_minus_one = static_cast<uint64_t>(shape[axis] - 1);
    uint64_t stride_magnitude   = 0;
    if (strides[axis] < 0)
    {
      if (strides[axis] == std::numeric_limits<int64_t>::min())
      {
        return false;
      }
      stride_magnitude = static_cast<uint64_t>(-strides[axis]);
    }
    else
    {
      stride_magnitude = static_cast<uint64_t>(strides[axis]);
    }

    if (stride_magnitude != 0 && extent_minus_one > std::numeric_limits<uint64_t>::max() / stride_magnitude)
    {
      return false;
    }
    const auto contribution = extent_minus_one * stride_magnitude;

    if (strides[axis] < 0)
    {
      if (contribution > std::numeric_limits<uint64_t>::max() - negative_offset)
      {
        return false;
      }
      negative_offset += contribution;
    }
    else
    {
      if (contribution > std::numeric_limits<uint64_t>::max() - positive_span)
      {
        return false;
      }
      positive_span += contribution;
    }
  }

  return negative_offset <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
      && negative_offset <= std::numeric_limits<uint64_t>::max() - positive_span;
}

bool layout_stride_strides_are_valid(size_t rank, const int64_t* shape, const int64_t* strides)
{
  if (strides == nullptr)
  {
    return false;
  }

  for (size_t axis = 0; axis < rank; ++axis)
  {
    if (strides[axis] <= 0)
    {
      return false;
    }
  }

  return strided_span_is_representable(rank, shape, strides);
}

bool contiguous_layout_strides_are_consistent(
  cccl_device_copy_layout_kind_t layout, size_t rank, const int64_t* shape, const int64_t* strides)
{
  if (!is_contiguous_layout(layout) || strides == nullptr)
  {
    return true;
  }

  auto check_axis = [&](size_t axis, uint64_t stride) {
    return shape[axis] <= 1
        || (stride <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
            && strides[axis] == static_cast<int64_t>(stride));
  };

  uint64_t stride = 1;
  if (layout == CCCL_DEVICE_COPY_LAYOUT_RIGHT)
  {
    for (size_t axis = rank; axis-- > 0;)
    {
      if (shape[axis] == 0)
      {
        return true;
      }
      if (!check_axis(axis, stride))
      {
        return false;
      }
      const auto extent = static_cast<uint64_t>(shape[axis]);
      if (stride > std::numeric_limits<uint64_t>::max() / extent)
      {
        return false;
      }
      stride *= extent;
    }
  }
  else
  {
    for (size_t axis = 0; axis < rank; ++axis)
    {
      if (shape[axis] == 0)
      {
        return true;
      }
      if (!check_axis(axis, stride))
      {
        return false;
      }
      const auto extent = static_cast<uint64_t>(shape[axis]);
      if (stride > std::numeric_limits<uint64_t>::max() / extent)
      {
        return false;
      }
      stride *= extent;
    }
  }

  return true;
}

CUresult validate_build_spec(cccl_device_copy_build_spec_t spec)
{
  if (spec.value_type.size == 0 || !is_power_of_two(spec.value_type.alignment))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((spec.value_type.size % spec.value_type.alignment) != 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (spec.rank == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_shape_metadata(spec.shape, spec.rank))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!validate_view_build(spec.source, spec.rank) || !validate_view_build(spec.destination, spec.rank))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  return CUDA_SUCCESS;
}

std::string dynamic_stride_template_arguments(size_t rank)
{
  std::string result;
  for (size_t axis = 0; axis < rank; ++axis)
  {
    if (axis != 0)
    {
      result += ", ";
    }
    result += "::cuda::dynamic_stride";
  }

  return result;
}

std::string casted_runtime_values(const char* values, size_t rank, const char* type)
{
  std::string result;
  for (size_t axis = 0; axis < rank; ++axis)
  {
    if (axis != 0)
    {
      result += ", ";
    }
    result += "static_cast<";
    result += type;
    result += ">(";
    result += values;
    result += "[";
    result += std::to_string(axis);
    result += "])";
  }

  return result;
}

const char* contiguous_layout_name(cccl_device_copy_layout_kind_t layout)
{
  return layout == CCCL_DEVICE_COPY_LAYOUT_LEFT ? "::cuda::std::layout_left" : "::cuda::std::layout_right";
}

std::string make_mdspan_view_source(
  const char* mdspan_type_name,
  const char* view_name,
  const char* data_name,
  const char* byte_offset_name,
  const char* strides_name,
  cccl_device_copy_layout_kind_t layout,
  bool is_const,
  size_t rank)
{
  const std::string value_type_name = is_const ? "const value_type" : "value_type";
  const std::string pointer_decl    = is_const ? "const auto* " : "auto* ";
  const std::string char_cast       = is_const ? "static_cast<const char*>(" : "static_cast<char*>(";
  const std::string pointer_cast    = "reinterpret_cast<" + value_type_name + "*>(";

  std::string src;
  src += "  " + pointer_decl + std::string(view_name) + "_effective =\n";
  src += "    " + pointer_cast + char_cast + data_name + ") + " + byte_offset_name + ");\n";

  if (is_layout_stride_relaxed(layout))
  {
    src += "  const runtime_strides_type " + std::string(view_name) + "_strides{"
         + casted_runtime_values(strides_name, rank, "offset_type") + "};\n";
    src += "  const offset_type " + std::string(view_name) + "_offset = __cccl_negative_stride_offset(extents, "
         + view_name + "_strides);\n";
    src += "  " + pointer_decl + std::string(view_name) + "_base = " + view_name + "_effective - " + view_name
         + "_offset;\n";
    src += "  using " + std::string(mdspan_type_name)
         + "_layout_type = cccl_device_copy_layout_stride_relaxed<runtime_strides_type, offset_type>;\n";
    src += "  using " + std::string(mdspan_type_name) + " = ::cuda::std::mdspan<" + value_type_name + ", extents_type, "
         + mdspan_type_name + "_layout_type>;\n";
    src += "  using " + std::string(view_name) + "_mapping_type = " + mdspan_type_name + "::mapping_type;\n";
    src += "  const " + std::string(mdspan_type_name) + " " + view_name + "{" + view_name + "_base, " + view_name
         + "_mapping_type{extents, " + view_name + "_strides, " + view_name + "_offset}};\n";
  }
  else if (is_layout_stride(layout))
  {
    src += "  const runtime_layout_stride_type " + std::string(view_name) + "_strides{"
         + casted_runtime_values(strides_name, rank, "index_type") + "};\n";
    src += "  using " + std::string(mdspan_type_name) + "_layout_type = ::cuda::std::layout_stride;\n";
    src += "  using " + std::string(mdspan_type_name) + " = ::cuda::std::mdspan<" + value_type_name + ", extents_type, "
         + mdspan_type_name + "_layout_type>;\n";
    src += "  using " + std::string(view_name) + "_mapping_type = " + mdspan_type_name + "::mapping_type;\n";
    src += "  const " + std::string(mdspan_type_name) + " " + view_name + "{" + view_name + "_effective, " + view_name
         + "_mapping_type{extents, " + view_name + "_strides}};\n";
  }
  else
  {
    src += "  using " + std::string(mdspan_type_name) + "_layout_type = " + contiguous_layout_name(layout) + ";\n";
    src += "  using " + std::string(mdspan_type_name) + " = ::cuda::std::mdspan<" + value_type_name + ", extents_type, "
         + mdspan_type_name + "_layout_type>;\n";
    src += "  const " + std::string(mdspan_type_name) + " " + view_name + "{" + view_name + "_effective, extents};\n";
  }

  src += "\n";
  return src;
}

std::string make_device_copy_source(cccl_device_copy_build_spec_t spec)
{
  std::string src = R"(#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__mdspan/layout_stride_relaxed.h>
#include <cuda/__mdspan/strides.h>
#include <cuda/std/array>
#include <cuda/std/mdspan>
#include <cuda/stream_ref>
#include <cub/device/device_copy.cuh>

#if _CCCL_HOSTJIT() && !_CCCL_HOSTED() && !defined(__CUDA_ARCH__) && !defined(_WIN32)
extern "C" void* dlopen(const char*, int);
extern "C" void* dlsym(void*, const char*);
#  ifndef RTLD_NOW
#    define RTLD_NOW 2
#  endif
#endif

static int __cccl_hostjit_init_cuda_driver()
{
  static int status = []() {
#if _CCCL_HOSTJIT() && !_CCCL_HOSTED()
#  if defined(__CUDA_ARCH__)
  return static_cast<int>(cudaSuccess);
#  elif defined(_WIN32)
  return static_cast<int>(cudaErrorNotSupported);
#  else
  if (::cuda::__driver::__getProcAddressFn() != nullptr)
  {
    return static_cast<int>(cudaSuccess);
  }

  static void* driver_library = ::dlopen("libcuda.so.1", RTLD_NOW);
  if (driver_library == nullptr)
  {
    return static_cast<int>(cudaErrorInitializationError);
  }

  static void* get_proc_address = ::dlsym(driver_library, "cuGetProcAddress_v2");
  if (get_proc_address == nullptr)
  {
    return static_cast<int>(cudaErrorInitializationError);
  }

  auto* stored_get_proc_address = ::cuda::__driver::__getProcAddressFn(
    reinterpret_cast<decltype(::cuGetProcAddress)*>(get_proc_address),
    true);
  if (stored_get_proc_address == nullptr)
  {
    return static_cast<int>(cudaErrorInitializationError);
  }
#  endif
#endif

  return static_cast<int>(cudaSuccess);
  }();
  return status;
}

)";

  src += "struct alignas(" + std::to_string(spec.value_type.alignment) + ") cccl_device_copy_value_t\n";
  src += "{\n";
  src += "  char data[" + std::to_string(spec.value_type.size) + "];\n";
  src += "};\n";
  src += "static_assert(sizeof(cccl_device_copy_value_t) == " + std::to_string(spec.value_type.size) + ");\n\n";

  src += R"(using cccl_device_copy_offset_t = long long;

template <class StridesT, class OffsetT>
struct cccl_device_copy_layout_stride_relaxed
{
  template <class ExtentsT>
  class mapping : public ::cuda::layout_stride_relaxed::mapping<ExtentsT, StridesT, OffsetT>
  {
    using base_type = ::cuda::layout_stride_relaxed::mapping<ExtentsT, StridesT, OffsetT>;

  public:
    using extents_type = typename base_type::extents_type;
    using index_type   = typename base_type::index_type;
    using size_type    = typename base_type::size_type;
    using rank_type    = typename base_type::rank_type;
    using layout_type  = cccl_device_copy_layout_stride_relaxed<StridesT, OffsetT>;

    using base_type::base_type;
    mapping() = default;

    friend constexpr bool operator==(const mapping& lhs, const mapping& rhs) noexcept
    {
      return static_cast<const base_type&>(lhs) == static_cast<const base_type&>(rhs);
    }

    friend constexpr bool operator!=(const mapping& lhs, const mapping& rhs) noexcept
    {
      return !(lhs == rhs);
    }
  };
};

template <class ExtentsT, class StridesT>
cccl_device_copy_offset_t __cccl_negative_stride_offset(const ExtentsT& extents, const StridesT& strides)
{
  cccl_device_copy_offset_t offset = 0;
  for (typename ExtentsT::rank_type axis = 0; axis < ExtentsT::rank(); ++axis)
  {
    const auto extent = extents.extent(axis);
    if (extent == 0)
    {
      return 0;
    }

    const auto stride = static_cast<cccl_device_copy_offset_t>(strides.stride(axis));
    if (stride < 0)
    {
      offset += static_cast<cccl_device_copy_offset_t>(extent - 1) * -stride;
    }
  }
  return offset;
}

extern "C" _CCCL_VISIBILITY_EXPORT int cccl_jit_device_copy(
  const void* source_data,
  unsigned long long source_byte_offset,
  const int64_t* source_shape,
  const int64_t* source_strides,
  void* destination_data,
  unsigned long long destination_byte_offset,
  const int64_t* destination_shape,
  const int64_t* destination_strides,
  void* stream)
{
  const int init_status = __cccl_hostjit_init_cuda_driver();
  if (init_status != static_cast<int>(cudaSuccess))
  {
    return init_status;
  }

)";

  const auto rank = spec.rank;
  src += "  (void) destination_shape;\n";
  if (!has_runtime_metadata(spec.shape, rank))
  {
    src += "  (void) source_shape;\n";
  }
  if (!is_strided_layout(spec.source.layout))
  {
    src += "  (void) source_strides;\n";
  }
  if (!is_strided_layout(spec.destination.layout))
  {
    src += "  (void) destination_strides;\n";
  }
  src += "\n";
  src += "  using value_type   = cccl_device_copy_value_t;\n";
  src += "  using index_type   = unsigned long long;\n";
  src += "  using offset_type  = cccl_device_copy_offset_t;\n";
  src += "  using extents_type = ::cuda::std::extents<index_type, "
       + cccl::detail::device_copy_codegen::extents_template_arguments(spec.shape, rank) + ">;\n";
  src += "  using runtime_strides_type = ::cuda::strides<offset_type, " + dynamic_stride_template_arguments(rank)
       + ">;\n";
  src += "  using runtime_layout_stride_type = ::cuda::std::array<index_type, " + std::to_string(rank) + ">;\n";
  src += "\n";
  src += "  const extents_type extents{"
       + cccl::detail::device_copy_codegen::dynamic_extent_constructor_arguments(
           "source_shape", spec.shape, rank, "index_type")
       + "};\n\n";
  src += make_mdspan_view_source(
    "input_type", "source_view", "source_data", "source_byte_offset", "source_strides", spec.source.layout, true, rank);
  src += make_mdspan_view_source(
    "output_type",
    "destination_view",
    "destination_data",
    "destination_byte_offset",
    "destination_strides",
    spec.destination.layout,
    false,
    rank);
  src += R"(  return static_cast<int>(
    cub::DeviceCopy::Copy(
      source_view,
      destination_view,
      ::cuda::stream_ref{reinterpret_cast<cudaStream_t>(stream)}));
}
)";

  return src;
}
} // namespace

CUresult cccl_device_copy_build_ex(
  cccl_device_copy_build_result_t* build_ptr,
  cccl_device_copy_build_spec_t spec,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* build_config)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *build_ptr = {};

  if (CUresult status = validate_build_spec(spec); status != CUDA_SUCCESS)
  {
    return status;
  }

  auto retained_shape               = retain_axis_metadata(spec.shape, spec.rank);
  auto retained_source_strides      = retain_stride_metadata(spec.source, spec.rank);
  auto retained_destination_strides = retain_stride_metadata(spec.destination, spec.rank);

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(build_config, cub_path, thrust_path);

  auto jit_config = hostjit::codegen::CubCall::make_jit_config(
    cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path, device_copy_fn_name);
  auto source          = make_device_copy_source(spec);
  auto retained_source = retain_source(source);

  if (const char* dump_path = std::getenv("CUBCALL_DUMP_SOURCE"))
  {
    std::ofstream f(dump_path);
    f << source;
  }

  auto compiler = std::make_unique<hostjit::JITCompiler>(jit_config);
  if (!compiler->compile(source))
  {
    throw std::runtime_error("DeviceCopy HostJIT compilation failed: " + compiler->getLastError());
  }

  auto fn = compiler->getFunction<device_copy_fn_t>(device_copy_fn_name);
  if (fn == nullptr)
  {
    throw std::runtime_error("DeviceCopy HostJIT function lookup failed: " + compiler->getLastError());
  }

  cccl::detail::copy_cubin(compiler->getCubin(), build_ptr->payload, build_ptr->payload_size);
  build_ptr->cc                  = cc_major * 10 + cc_minor;
  build_ptr->source              = retained_source.release();
  build_ptr->source_size         = source.size();
  build_ptr->jit_compiler        = compiler.release();
  build_ptr->copy_fn             = reinterpret_cast<void*>(fn);
  build_ptr->value_type          = spec.value_type;
  build_ptr->rank                = spec.rank;
  build_ptr->shape               = retained_shape.release();
  build_ptr->source_strides      = retained_source_strides.release();
  build_ptr->destination_strides = retained_destination_strides.release();
  build_ptr->source_layout       = spec.source.layout;
  build_ptr->destination_layout  = spec.destination.layout;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  std::fprintf(stderr, "\nEXCEPTION in cccl_device_copy_build_ex(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_copy_build(
  cccl_device_copy_build_result_t* build_ptr,
  cccl_device_copy_build_spec_t spec,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_copy_build_ex(
    build_ptr, spec, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}

CUresult cccl_device_copy(cccl_device_copy_build_result_t build,
                          cccl_device_copy_source_view_t source,
                          cccl_device_copy_destination_view_t destination,
                          CUstream stream)
try
{
  if (build.copy_fn == nullptr || build.rank == 0 || !is_supported_layout(build.source_layout)
      || !is_supported_layout(build.destination_layout))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (build.shape == nullptr || source.shape == nullptr || destination.shape == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!valid_shape_metadata(build.shape, build.rank))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!validate_view_build(cccl_device_copy_view_build_t{build.source_layout, build.source_strides}, build.rank)
      || !validate_view_build(cccl_device_copy_view_build_t{build.destination_layout, build.destination_strides},
                              build.rank))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (size_t axis = 0; axis < build.rank; ++axis)
  {
    if (source.shape[axis] < 0 || destination.shape[axis] < 0 || source.shape[axis] != destination.shape[axis])
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (build.shape[axis].kind == CCCL_DEVICE_COPY_AXIS_STATIC && source.shape[axis] != build.shape[axis].value)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }
  if (!product_is_representable(build.rank, source.shape))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const bool source_strides_are_consistent =
    contiguous_layout_strides_are_consistent(build.source_layout, build.rank, source.shape, source.strides);
  const bool destination_strides_are_consistent = contiguous_layout_strides_are_consistent(
    build.destination_layout, build.rank, destination.shape, destination.strides);
  if (!source_strides_are_consistent || !destination_strides_are_consistent)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (is_layout_stride(build.source_layout)
      && !layout_stride_strides_are_valid(build.rank, source.shape, source.strides))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (is_layout_stride(build.destination_layout)
      && !layout_stride_strides_are_valid(build.rank, destination.shape, destination.strides))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (is_layout_stride_relaxed(build.source_layout)
      && (source.strides == nullptr || !strided_span_is_representable(build.rank, source.shape, source.strides)))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (is_layout_stride_relaxed(build.destination_layout)
      && (destination.strides == nullptr
          || !strided_span_is_representable(build.rank, destination.shape, destination.strides)))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (!effective_address_is_aligned(source.data, source.byte_offset, build.value_type.alignment)
      || !effective_address_is_aligned(destination.data, destination.byte_offset, build.value_type.alignment))
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto fn          = reinterpret_cast<device_copy_fn_t>(build.copy_fn);
  const int status = fn(
    source.data,
    static_cast<unsigned long long>(source.byte_offset),
    source.shape,
    source.strides,
    destination.data,
    static_cast<unsigned long long>(destination.byte_offset),
    destination.shape,
    destination.strides,
    reinterpret_cast<void*>(stream));

  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  std::fprintf(stderr, "\nEXCEPTION in cccl_device_copy(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_copy_cleanup(cccl_device_copy_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  delete[] build_ptr->destination_strides;
  build_ptr->destination_strides = nullptr;
  delete[] build_ptr->source_strides;
  build_ptr->source_strides = nullptr;
  delete[] build_ptr->shape;
  build_ptr->shape = nullptr;
  delete[] build_ptr->source;
  build_ptr->source      = nullptr;
  build_ptr->source_size = 0;
  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->copy_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  std::fprintf(stderr, "\nEXCEPTION in cccl_device_copy_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
