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
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_device.cuh>

#include <format>
#include <type_traits>
#include <vector>

#include <cccl/c/binary_search.h>
#include <cccl/c/types.h>
#include <for/for_op_helper.h>
#include <jit_templates/templates/input_iterator.h>
#include <jit_templates/templates/operation.h>
#include <jit_templates/templates/output_iterator.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/indirect_arg.h>
#include <util/types.h>

struct op_wrapper;
struct device_reduce_policy;

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be size_t");

static cudaError_t Invoke(
  indirect_arg_t d_in,
  size_t num_items,
  indirect_arg_t d_values,
  size_t num_values,
  indirect_arg_t d_out,
  cccl_op_t op,
  int /*cc*/,
  CUfunction kernel,
  CUstream stream)
{
  cudaError error = cudaSuccess;

  if (num_values == 0)
  {
    return error;
  }

  void* args[] = {&d_in, &num_items, &d_values, &num_values, &d_out, &op};

  const unsigned int thread_count = 256;
  const size_t items_per_block    = 512;
  const size_t block_sz           = cuda::ceil_div(num_values, items_per_block);

  if (block_sz > std::numeric_limits<unsigned int>::max())
  {
    return cudaErrorInvalidValue;
  }
  const unsigned int block_count = static_cast<unsigned int>(block_sz);

  check(cuLaunchKernel(kernel, block_count, 1, 1, thread_count, 1, 1, 0, stream, args, 0));

  // Check for failure to launch
  error = CubDebug(cudaPeekAtLastError());

  return error;
}

struct binary_search_data_iterator_tag;
struct binary_search_values_iterator_tag;
struct binary_search_output_iterator_tag;
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

  const char* name = "test";

  const int cc = cc_major * 10 + cc_minor;

  auto [d_data_it_name, d_data_it_src] =
    get_specialization<binary_search_data_iterator_tag>(template_id<input_iterator_traits>(), d_data);
  auto [d_values_it_name, d_values_it_src] =
    get_specialization<binary_search_values_iterator_tag>(template_id<input_iterator_traits>(), d_values);
  auto [d_out_it_name, d_out_it_src] = get_specialization<binary_search_output_iterator_tag>(
    template_id<output_iterator_traits>(), d_out, d_out.value_type);
  auto [op_name, op_src] =
    get_specialization<binary_search_op_tag>(template_id<binary_user_predicate_traits>(), op, d_data.value_type);

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

  const std::string src = std::format(
    R"XXX(
#include <cub/agent/agent_for.cuh>
#include <cub/detail/binary_search_helpers.cuh>
#include <cuda/__iterator/zip_iterator.h>

{11}

struct __align__({10}) storage_t {{
  char data[{9}];
}};

{0}
{2}
{4}
{6}

using policy_dim_t = cub::detail::for_each::policy_t<256, 2>;
using OffsetT = cuda::std::size_t;

struct device_for_policy
{{
  struct ActivePolicy
  {{
    using for_policy_t = policy_dim_t;
  }};
}};

CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(device_for_policy::ActivePolicy::for_policy_t::block_threads)
void binary_search_kernel({1} d_data, OffsetT num_data, {3} d_values, OffsetT num_values, {5} d_out, {7} op)
{{
  auto d_out_typed = [&] {{
    constexpr auto out_is_ptr = cuda::std::is_pointer_v<decltype(d_out)>;
    constexpr auto out_matches_items = cuda::std::is_same_v<decltype(*d_out), decltype(d_data)>;
    constexpr auto need_cast = out_is_ptr && !out_matches_items;

    if constexpr (need_cast) {{
      static_assert(sizeof(decltype(*d_out)) == sizeof(decltype(d_data)), "");
      static_assert(alignof(decltype(*d_out)) == alignof(decltype(d_data)), "");
      return reinterpret_cast<{1} *>(d_out);
    }}
    else {{
      return d_out;
    }}
  }}();

  auto input_it     = cuda::make_zip_iterator(d_values, d_out_typed);
  auto comp_wrapper = cub::detail::find::make_comp_wrapper<{8}>(d_data, d_data + num_data, op);
  auto agent_op     = [&comp_wrapper, &input_it](OffsetT index) {{
    comp_wrapper(input_it[index]);
  }};

  using active_policy_t = device_for_policy::ActivePolicy::for_policy_t;
  using agent_t         = cub::detail::for_each::agent_block_striped_t<active_policy_t, OffsetT, decltype(agent_op)>;

  constexpr auto block_threads  = active_policy_t::block_threads;
  constexpr auto items_per_tile = active_policy_t::items_per_thread * block_threads;

  const auto tile_base     = static_cast<OffsetT>(blockIdx.x) * items_per_tile;
  const auto num_remaining = num_values - tile_base;
  const auto items_in_tile = static_cast<OffsetT>(num_remaining < items_per_tile ? num_remaining : items_per_tile);

  if (items_in_tile == items_per_tile)
  {{
    agent_t{{tile_base, agent_op}}.template consume_tile<true>(items_per_tile, block_threads);
  }}
  else
  {{
    agent_t{{tile_base, agent_op}}.template consume_tile<false>(items_in_tile, block_threads);
  }}
}}
)XXX",
    d_data_it_src,
    d_data_it_name,
    d_values_it_src,
    d_values_it_name,
    d_out_it_src,
    d_out_it_name,
    op_src,
    op_name,
    mode_t,
    d_out.value_type.size,
    d_out.value_type.alignment,
    jit_template_header_contents);

  const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

  std::vector<const char*> args = {
    arch.c_str(),
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    "-std=c++20",
    "-rdc=true",
    "-dlto",
    "-DCUB_DISABLE_CDP"};

  cccl::detail::extend_args_with_build_config(args, config);

  constexpr size_t num_lto_args   = 2;
  const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

  std::string lowered_name;

  // Collect all LTO-IRs to be linked
  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};

  appender.append_operation(op);

  // Add iterator definitions if present
  for (const auto& it_type : {d_data, d_values, d_out})
  {
    if (cccl_iterator_kind_t::CCCL_ITERATOR == it_type.type)
    {
      appender.append_operation(it_type.advance);
      appender.append_operation(it_type.dereference);
    }
  }

  nvrtc_link_result result =
    begin_linking_nvrtc_program(num_lto_args, lopts)
      ->add_program(nvrtc_translation_unit{src, name})
      ->add_expression({"binary_search_kernel"})
      ->compile_program({args.data(), args.size()})
      ->get_name({"binary_search_kernel", lowered_name})
      ->link_program()
      ->add_link_list(linkable_list)
      ->finalize_program();

  cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  check(cuLibraryGetKernel(&build_ptr->kernel, build_ptr->library, lowered_name.c_str()));

  build_ptr->cc         = cc;
  build_ptr->cubin      = (void*) result.data.release();
  build_ptr->cubin_size = result.size;

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
    auto exec_status =
      Invoke(d_data, num_items, d_values, num_values, d_out, op, build.cc, (CUfunction) build.kernel, stream);
    error = static_cast<CUresult>(exec_status);
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

  std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
  check(cuLibraryUnload(build_ptr->library));

  return CUDA_SUCCESS;
}
catch (...)
{
  return CUDA_ERROR_UNKNOWN;
}
