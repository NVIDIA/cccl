//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>

#include <algorithm>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <cccl/c/binary_search.h>
#include <cccl/c/transform.h>
#include <cccl/c/types.h>
#include <jit_templates/templates/input_iterator.h>
#include <jit_templates/templates/operation.h>
#include <nvrtc/command_list.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/types.h>

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

namespace binary_search
{
struct op_state_header_t
{
  void* data;
  OffsetT num_data;
};

struct op_state_t
{
  std::unique_ptr<char[]> storage;

  void* get()
  {
    return storage.get();
  }
};

static size_t align_up(size_t offset, size_t alignment)
{
  const size_t remainder = offset % alignment;
  return remainder == 0 ? offset : offset + alignment - remainder;
}

static size_t comparator_state_offset(cccl_op_t op)
{
  return op.type == CCCL_STATEFUL ? align_up(sizeof(op_state_header_t), op.alignment) : sizeof(op_state_header_t);
}

static size_t op_state_alignment(cccl_op_t op)
{
  return op.type == CCCL_STATEFUL ? std::max(alignof(op_state_header_t), op.alignment) : alignof(op_state_header_t);
}

static size_t op_state_size(cccl_op_t op)
{
  const size_t unaligned_size =
    op.type == CCCL_STATEFUL ? comparator_state_offset(op) + op.size : sizeof(op_state_header_t);
  return align_up(unaligned_size, op_state_alignment(op));
}

static op_state_t make_op_state(cccl_iterator_t data, OffsetT num_data, cccl_op_t op)
{
  op_state_t result{std::make_unique<char[]>(op_state_size(op))};
  char* raw = static_cast<char*>(result.get());

  auto* header     = reinterpret_cast<op_state_header_t*>(raw);
  header->data     = data.state;
  header->num_data = num_data;

  if (op.type == CCCL_STATEFUL)
  {
    std::memcpy(raw + comparator_state_offset(op), op.state, op.size);
  }

  return result;
}
} // namespace binary_search

static CUresult Invoke(
  cccl_iterator_t d_in,
  uint64_t num_items,
  cccl_iterator_t d_values,
  uint64_t num_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_device_binary_search_build_result_t build,
  CUstream stream)
{
  auto state = binary_search::make_op_state(d_in, static_cast<OffsetT>(num_items), op);

  cccl_op_t transform_op = op;
  transform_op.type      = CCCL_STATEFUL;
  transform_op.state     = state.get();
  transform_op.size      = build.op_state_size;
  transform_op.alignment = build.op_state_alignment;

  return cccl_device_unary_transform(build.transform, d_values, d_out, num_values, transform_op, stream);
}

struct binary_search_data_iterator_tag;
struct binary_search_values_iterator_tag;
struct binary_search_op_tag;

CUresult cccl_device_binary_search_build_ex(
  cccl_device_binary_search_build_result_t* build_ptr,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  if (d_data.type == cccl_iterator_kind_t::CCCL_ITERATOR)
  {
    throw std::runtime_error(std::string("Iterators are unsupported in for_each currently"));
  }

  auto [d_data_it_name, d_data_it_src] =
    get_specialization<binary_search_data_iterator_tag>(template_id<input_iterator_traits>(), d_data);
  auto [op_name, op_src] = get_specialization<binary_search_op_tag>(
    template_id<binary_user_predicate_traits>(), op, d_data.value_type, d_data.value_type);

  const std::string mode_t = [&] {
    switch (mode)
    {
      case CCCL_BINARY_SEARCH_LOWER_BOUND:
        return "cub::detail::find::lower_bound";
      case CCCL_BINARY_SEARCH_UPPER_BOUND:
        return "cub::detail::find::upper_bound";
    }
    throw std::runtime_error(std::format("Invalid binary search mode ({})", static_cast<int>(mode)));
  }();

  const bool user_defined_comparator = op.type == CCCL_STATEFUL || op.type == CCCL_STATELESS;
  const bool comparator_stateful     = op.type == CCCL_STATEFUL;
  const auto comparator_offset       = binary_search::comparator_state_offset(op);
  const std::string output_t         = cccl_type_enum_to_name(d_out.value_type.type);
  const size_t storage_size = std::max({d_data.value_type.size, d_values.value_type.size, d_out.value_type.size});
  const size_t storage_alignment =
    std::max({d_data.value_type.alignment, d_values.value_type.alignment, d_out.value_type.alignment});

  const std::string transform_op_src = std::format(
    R"XXX(
#include <cub/detail/binary_search_helpers.cuh>
#include <cuda/std/__cstring/memcpy.h>

{8}
struct __align__({7}) storage_t {{
  char data[{6}];
}};

{0}
{2}
using OffsetT = cuda::std::size_t;

struct binary_search_op_state
{{
  {1} data;
  OffsetT num_data;
}};

extern "C" __device__ void binary_search_transform_op(void* state, const void* value, void* result)
{{
  auto* header     = static_cast<binary_search_op_state*>(state);
  const auto& item = *static_cast<const {3}*>(value);

  {4} comparator{{}};
  if constexpr ({9})
  {{
    ::cuda::std::memcpy(&comparator, static_cast<char*>(state) + {10}, sizeof(comparator));
  }}

  const auto search_op =
    cub::detail::find::make_binary_search_transform_op<{11}>(header->data, header->num_data, comparator);
  *static_cast<{5}*>(result) =
    static_cast<{5}>(search_op(item));
}}
)XXX",
    d_data_it_src,
    d_data_it_name,
    op_src,
    cccl_type_enum_to_name(d_values.value_type.type),
    op_name,
    output_t,
    storage_size,
    storage_alignment,
    jit_template_header_contents,
    comparator_stateful ? "true" : "false",
    comparator_offset,
    mode_t);

  cccl_op_t transform_op = op;
  transform_op.type      = CCCL_STATEFUL;
  transform_op.name      = "binary_search_transform_op";
  transform_op.code      = transform_op_src.c_str();
  transform_op.code_size = transform_op_src.size();
  transform_op.code_type = CCCL_OP_CPP_SOURCE;
  transform_op.size      = binary_search::op_state_size(op);
  transform_op.alignment = binary_search::op_state_alignment(op);

  std::vector<const char*> extra_ltoirs;
  std::vector<size_t> extra_ltoir_sizes;
  std::unique_ptr<char[]> comparator_ltoir;

  const std::string arch        = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);
  std::vector<const char*> args = {
    arch.c_str(),
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    "-rdc=true",
    "-dlto",
    "-default-device",
    "-DCUB_DISABLE_CDP",
    "-std=c++20"};
  cccl::detail::extend_args_with_build_config(args, config);

  constexpr size_t num_lto_args   = 2;
  const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

  if (user_defined_comparator && op.code_type == CCCL_OP_CPP_SOURCE && op.code_size != 0)
  {
    auto [lto_size, lto_buf] =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit{op.code, op.name})
        ->compile_program({args.data(), args.size()})
        ->get_program_ltoir();
    comparator_ltoir = std::move(lto_buf);
    extra_ltoirs.push_back(comparator_ltoir.get());
    extra_ltoir_sizes.push_back(lto_size);
  }
  else if (user_defined_comparator && op.code_type == CCCL_OP_LTOIR && op.code_size != 0)
  {
    extra_ltoirs.push_back(op.code);
    extra_ltoir_sizes.push_back(op.code_size);
  }
  for (size_t i = 0; user_defined_comparator && i < op.num_extra_ltoirs; ++i)
  {
    extra_ltoirs.push_back(op.extra_ltoirs[i]);
    extra_ltoir_sizes.push_back(op.extra_ltoir_sizes[i]);
  }
  transform_op.extra_ltoirs      = extra_ltoirs.data();
  transform_op.extra_ltoir_sizes = extra_ltoir_sizes.data();
  transform_op.num_extra_ltoirs  = extra_ltoirs.size();

  check(cccl_device_unary_transform_build_ex(
    &build_ptr->transform,
    d_values,
    d_out,
    transform_op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    config));

  build_ptr->op_state_size      = transform_op.size;
  build_ptr->op_state_alignment = transform_op.alignment;

  return CUDA_SUCCESS;
}
catch (...)
{
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_binary_search(
  cccl_device_binary_search_build_result_t build,
  cccl_iterator_t d_data,
  uint64_t num_items,
  cccl_iterator_t d_values,
  uint64_t num_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;

  try
  {
    pushed = try_push_context();
    error  = Invoke(d_data, num_items, d_values, num_values, d_out, op, build, stream);
  }
  catch (...)
  {
    error = CUDA_ERROR_UNKNOWN;
  }

  if (pushed)
  {
    CUcontext dummy;
    cuCtxPopCurrent(&dummy);
  }

  return error;
}

CUresult cccl_device_binary_search_build(
  cccl_device_binary_search_build_result_t* build,
  cccl_binary_search_mode_t mode,
  cccl_iterator_t d_data,
  cccl_iterator_t d_values,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_binary_search_build_ex(
    build,
    mode,
    d_data,
    d_values,
    d_out,
    op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_binary_search_cleanup(cccl_device_binary_search_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  return cccl_device_transform_cleanup(&build_ptr->transform);
}
catch (...)
{
  return CUDA_ERROR_UNKNOWN;
}
