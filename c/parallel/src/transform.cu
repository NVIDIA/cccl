//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/dispatch/dispatch_transform.cuh>
#include <cub/device/dispatch/tuning/tuning_transform.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/memory>

#include <format>
#include <string>
#include <type_traits>
#include <vector>

#include <stdio.h> // printf

#include <cccl/c/transform.h>
#include <cccl/c/types.h> // cccl_type_info
#include <kernels/iterators.h>
#include <kernels/operators.h>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>
#include <util/context.h>
#include <util/errors.h>
#include <util/indirect_arg.h>
#include <util/types.h>

struct op_wrapper;
struct device_transform_policy;

using OffsetT = ptrdiff_t;
static_assert(std::is_same_v<cub::detail::choose_signed_offset_t<OffsetT>, OffsetT>,
              "OffsetT must be signed int32 or int64");

struct input_storage_t;
struct input1_storage_t;
struct input2_storage_t;
struct output_storage_t;

namespace transform
{
constexpr auto input_iterator_name  = "input_iterator_t";
constexpr auto input1_iterator_name = "input1_iterator_t";
constexpr auto input2_iterator_name = "input2_iterator_t";
constexpr auto output_iterator_name = "output_iterator_t";

template <typename StorageT>
const std::string get_iterator_name(cccl_iterator_t iterator, const std::string& name)
{
  if (iterator.type == cccl_iterator_kind_t::CCCL_POINTER)
  {
    return cccl_type_enum_to_name<StorageT>(iterator.value_type.type, true);
  }
  return name;
}

std::string get_kernel_name(cccl_iterator_t input_it, cccl_iterator_t output_it, cccl_op_t /*op*/)
{
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_transform_policy>(&chained_policy_t));

  const std::string input_iterator_t  = get_iterator_name<input_storage_t>(input_it, input_iterator_name);
  const std::string output_iterator_t = get_iterator_name<output_storage_t>(output_it, output_iterator_name);

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(cccl_type_name_from_nvrtc<op_wrapper>(&transform_op_t));

  return std::format(
    "cub::detail::transform::transform_kernel<{0}, {1}, cub::detail::transform::always_true_predicate, {2}, {3}, {4}>",
    chained_policy_t, // 0
    offset_t, // 1
    transform_op_t, // 2
    output_iterator_t, // 3
    input_iterator_t); // 4
}

std::string
get_kernel_name(cccl_iterator_t input1_it, cccl_iterator_t input2_it, cccl_iterator_t output_it, cccl_op_t /*op*/)
{
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_transform_policy>(&chained_policy_t));

  const std::string input1_iterator_t = get_iterator_name<input1_storage_t>(input1_it, input1_iterator_name);
  const std::string input2_iterator_t = get_iterator_name<input2_storage_t>(input2_it, input2_iterator_name);
  const std::string output_iterator_t = get_iterator_name<output_storage_t>(output_it, output_iterator_name);

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(cccl_type_name_from_nvrtc<op_wrapper>(&transform_op_t));

  return std::format(
    "cub::detail::transform::transform_kernel<{0}, {1}, cub::detail::transform::always_true_predicate, {2}, {3}, {4}, "
    "{5}>",
    chained_policy_t, // 0
    offset_t, // 1
    transform_op_t, // 2
    output_iterator_t, // 3
    input1_iterator_t, // 4
    input2_iterator_t); // 5
}

namespace cdt = cub::detail::transform;

struct cache
{
  cuda::std::optional<cub::detail::transform::cuda_expected<cub::detail::transform::async_config>> async_config{};
  cuda::std::optional<cub::detail::transform::cuda_expected<cub::detail::transform::prefetch_config>> prefetch_config{};
};

template <int NumInputs>
struct transform_kernel_source
{
  cccl_device_transform_build_result_t& build;
  cuda::std::array<cub::detail::iterator_info, NumInputs> inputs;

  template <class ActionT>
  cub::detail::transform::cuda_expected<cub::detail::transform::async_config>
  CacheAsyncConfiguration(const ActionT& action)
  {
    auto cache = reinterpret_cast<transform::cache*>(build.cache);
    if (!cache->async_config.has_value())
    {
      cache->async_config = action();
    }
    return *cache->async_config;
  }

  template <class ActionT>
  cub::detail::transform::cuda_expected<cub::detail::transform::prefetch_config>
  CachePrefetchConfiguration(const ActionT& action)
  {
    auto cache = reinterpret_cast<transform::cache*>(build.cache);
    if (!cache->prefetch_config.has_value())
    {
      cache->prefetch_config = action();
    }
    return *cache->prefetch_config;
  }

  CUkernel TransformKernel() const
  {
    return build.transform_kernel;
  }

  int LoadedBytesPerIteration() const
  {
    return build.loaded_bytes_per_iteration;
  }

  const auto& InputIteratorInfos() const
  {
    return inputs;
  }

  template <typename It>
  static constexpr It MakeIteratorKernelArg(It it)
  {
    return it;
  }

  static cdt::kernel_arg<char*> MakeAlignedBasePtrKernelArg(indirect_iterator_t it, int align)
  {
    _CCCL_ASSERT(it.value_size != 0, "a non-pointer iterator passed into MakeALignedBasePtrKernelArg");
    return cdt::make_aligned_base_ptr_kernel_arg(*static_cast<char**>(it.ptr), align);
  }

private:
  static auto is_pointer_aligned(const indirect_iterator_t& it, ::cuda::std::size_t alignment)
  {
    return it.value_size != 0 && ::cuda::is_aligned(*static_cast<char**>(it.ptr), alignment);
  }

public:
  template <typename... Iterators>
  static bool CanVectorize(int vec_size, Iterators... its)
  {
    return (is_pointer_aligned(its, its.value_size * vec_size) && ...);
  }
};

auto make_iterator_info(cccl_iterator_t it) -> cub::detail::iterator_info
{
  // TODO(bgruber): CCCL_STORAGE is not necessarily trivially relocatable, but how can we know this here?
  // gevtushenko said, that he is not aware of types which are not trivially relocatable for now, since
  // CCCL_STORAGE is used to store user-defined types, and CCCL.C does not support any kind of constructors at the
  // moment. So I guess we are fine until CCCL_STORAGE supports such complex types.
  const auto vt_is_trivially_relocatable = true; // input_it.value_type.type != CCCL_STORAGE;
  const auto is_contiguous               = it.type == CCCL_POINTER;
  return {static_cast<int>(it.value_type.size),
          static_cast<int>(it.value_type.alignment),
          vt_is_trivially_relocatable,
          is_contiguous};
}
} // namespace transform

CUresult cccl_device_unary_transform_build_ex(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
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
  const char* name = "test";

  const auto input_it_value_t  = cccl_type_enum_to_name<input_storage_t>(input_it.value_type.type);
  const auto output_it_value_t = cccl_type_enum_to_name<output_storage_t>(output_it.value_type.type);
  const auto offset_t          = cccl_type_enum_to_name(cccl_type_enum::CCCL_INT64);
  const std::string input_iterator_src =
    make_kernel_input_iterator(offset_t, transform::input_iterator_name, input_it_value_t, input_it);
  const std::string output_iterator_src =
    make_kernel_output_iterator(offset_t, transform::output_iterator_name, output_it_value_t, output_it);
  const std::string op_src = make_kernel_user_unary_operator(input_it_value_t, output_it_value_t, op);

  const auto inputs     = cuda::std::array<cub::detail::iterator_info, 1>{transform::make_iterator_info(input_it)};
  const auto output     = transform::make_iterator_info(output_it);
  const auto policy_sel = cub::detail::transform::policy_selector<1>{false, true, inputs, output};

  // TODO(bgruber): drop this if tuning policies become formattable
  std::stringstream policy_sel_str;
  policy_sel_str << policy_sel(cuda::to_arch_id(cuda::compute_capability{cc_major, cc_minor}));

  const auto policy_hub_expr = std::format(
    "cub::detail::transform::policy_selector_from_types<false, true, ::cuda::std::tuple<{}>, {}>",
    transform::get_iterator_name<input_storage_t>(input_it, transform::input_iterator_name),
    transform::get_iterator_name<output_storage_t>(output_it, transform::output_iterator_name));

  std::string final_src = std::format(
    R"XXX(
#include <cub/device/dispatch/tuning/tuning_transform.cuh>
#include <cub/device/dispatch/kernels/kernel_transform.cuh>
struct __align__({1}) input_storage_t {{
  char data[{0}];
}};
struct __align__({3}) output_storage_t {{
  char data[{2}];
}};
{4}
{5}
{6}
using device_transform_policy = {7};
using namespace cub;
using namespace cub::detail::transform;
static_assert(device_transform_policy()(::cuda::arch_id{{CUB_PTX_ARCH / 10}}) == {8}, "Host generated and JIT compiled policy mismatch");
)XXX",
    input_it.value_type.size, // 0
    input_it.value_type.alignment, // 1
    output_it.value_type.size, // 2
    output_it.value_type.alignment, // 3
    input_iterator_src, // 4
    output_iterator_src, // 5
    op_src, // 6
    policy_hub_expr, // 7
    policy_sel_str.view()); // 8

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
    fflush(stdout);
#endif

  std::string kernel_name = transform::get_kernel_name(input_it, output_it, op);
  std::string kernel_lowered_name;

  const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

  // Note: `-default-device` is needed because of the use of lambdas
  // in the transform kernel code. Qualifying those explicitly with
  // `__device__` seems not to be supported by NVRTC.
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

  // Collect all LTO-IRs to be linked.
  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};

  appender.append_operation(op);
  appender.add_iterator_definition(input_it);
  appender.add_iterator_definition(output_it);

  nvrtc_link_result result =
    begin_linking_nvrtc_program(num_lto_args, lopts)
      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
      ->add_expression({kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({kernel_name, kernel_lowered_name})
      ->link_program()
      ->add_link_list(linkable_list)
      ->finalize_program();

  cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  check(cuLibraryGetKernel(&build_ptr->transform_kernel, build_ptr->library, kernel_lowered_name.c_str()));

  build_ptr->loaded_bytes_per_iteration = static_cast<int>(input_it.value_type.size);
  build_ptr->cc                         = cc_major * 10 + cc_minor;
  build_ptr->cubin                      = (void*) result.data.release();
  build_ptr->cubin_size                 = result.size;
  build_ptr->cache                      = new transform::cache();

  // avoid new and delete which requires the allocated and freed types to match
  static_assert(std::is_trivially_copyable_v<decltype(policy_sel)>);
  build_ptr->runtime_policy = std::malloc(sizeof(policy_sel));
  std::memcpy(build_ptr->runtime_policy, &policy_sel, sizeof(policy_sel));

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_unary_transform_build(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_unary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));
    error = static_cast<CUresult>(transform::cdt::dispatch<transform::cdt::requires_stable_address::no>(
      ::cuda::std::tuple<indirect_iterator_t>{d_in},
      indirect_iterator_t{d_out},
      static_cast<OffsetT>(num_items),
      transform::cdt::always_true_predicate{},
      op,
      stream,
      *static_cast<cub::detail::transform::policy_selector<1>*>(build.runtime_policy),
      transform::transform_kernel_source<1>{build, {transform::make_iterator_info(d_in)}},
      cub::detail::CudaDriverLauncherFactory{cu_device, build.cc}));
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_unary_transform(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }
  if (pushed)
  {
    CUcontext cu_context;
    cuCtxPopCurrent(&cu_context);
  }
  return error;
}

CUresult cccl_device_binary_transform_build_ex(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t input1_it,
  cccl_iterator_t input2_it,
  cccl_iterator_t output_it,
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
  const char* name = "test";

  const auto input1_it_value_t = cccl_type_enum_to_name<input1_storage_t>(input1_it.value_type.type);
  const auto input2_it_value_t = cccl_type_enum_to_name<input2_storage_t>(input2_it.value_type.type);

  const auto output_it_value_t = cccl_type_enum_to_name<output_storage_t>(output_it.value_type.type);
  const auto offset_t          = cccl_type_enum_to_name(cccl_type_enum::CCCL_INT64);
  const std::string input1_iterator_src =
    make_kernel_input_iterator(offset_t, transform::input1_iterator_name, input1_it_value_t, input1_it);
  const std::string input2_iterator_src =
    make_kernel_input_iterator(offset_t, transform::input2_iterator_name, input2_it_value_t, input2_it);

  const std::string output_iterator_src =
    make_kernel_output_iterator(offset_t, transform::output_iterator_name, output_it_value_t, output_it);
  const std::string op_src =
    make_kernel_user_binary_operator(input1_it_value_t, input2_it_value_t, output_it_value_t, op);

  const auto inputs = cuda::std::array<cub::detail::iterator_info, 2>{
    transform::make_iterator_info(input1_it), transform::make_iterator_info(input2_it)};
  const auto output     = transform::make_iterator_info(output_it);
  const auto policy_sel = cub::detail::transform::policy_selector<2>{false, true, inputs, output};

  // TODO(bgruber): drop this if tuning policies become formattable
  std::stringstream policy_sel_str;
  policy_sel_str << policy_sel(cuda::to_arch_id(cuda::compute_capability{cc_major, cc_minor}));

  const auto policy_hub_expr = std::format(
    "cub::detail::transform::policy_selector_from_types<false, true, ::cuda::std::tuple<{0}, {1}>, {2}>",
    transform::get_iterator_name<input1_storage_t>(input1_it, transform::input1_iterator_name),
    transform::get_iterator_name<input2_storage_t>(input2_it, transform::input2_iterator_name),
    transform::get_iterator_name<output_storage_t>(output_it, transform::output_iterator_name));

  std::string final_src = std::format(
    R"XXX(
#include <cub/device/dispatch/kernels/kernel_transform.cuh>
struct __align__({1}) input1_storage_t {{
  char data[{0}];
}};
struct __align__({3}) input2_storage_t {{
  char data[{2}];
}};

struct __align__({5}) output_storage_t {{
  char data[{4}];
}};
{6}
{7}
{8}
{9}
using device_transform_policy = {10};
using namespace cub;
using namespace cub::detail::transform;
static_assert(device_transform_policy()(::cuda::arch_id{{CUB_PTX_ARCH / 10}}) == {11}, "Host generated and JIT compiled policy mismatch");
)XXX",
    input1_it.value_type.size, // 0
    input1_it.value_type.alignment, // 1
    input2_it.value_type.size, // 2
    input2_it.value_type.alignment, // 3
    output_it.value_type.size, // 4
    output_it.value_type.alignment, // 5
    input1_iterator_src, // 6
    input2_iterator_src, // 7
    output_iterator_src, // 8
    op_src, // 9
    policy_hub_expr, // 10
    policy_sel_str.view()); // 11

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", final_src.c_str());
    fflush(stdout);
#endif

  std::string kernel_name = transform::get_kernel_name(input1_it, input2_it, output_it, op);
  std::string kernel_lowered_name;

  const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

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

  // Collect all LTO-IRs to be linked.
  nvrtc_linkable_list linkable_list;
  nvrtc_linkable_list_appender appender{linkable_list};

  appender.append_operation(op);
  appender.add_iterator_definition(input1_it);
  appender.add_iterator_definition(input2_it);
  appender.add_iterator_definition(output_it);

  nvrtc_link_result result =
    begin_linking_nvrtc_program(num_lto_args, lopts)
      ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
      ->add_expression({kernel_name})
      ->compile_program({args.data(), args.size()})
      ->get_name({kernel_name, kernel_lowered_name})
      ->link_program()
      ->add_link_list(linkable_list)
      ->finalize_program();

  cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  check(cuLibraryGetKernel(&build_ptr->transform_kernel, build_ptr->library, kernel_lowered_name.c_str()));

  build_ptr->loaded_bytes_per_iteration = static_cast<int>((input1_it.value_type.size + input2_it.value_type.size));
  build_ptr->cc                         = cc_major * 10 + cc_minor;
  build_ptr->cubin                      = (void*) result.data.release();
  build_ptr->cubin_size                 = result.size;
  build_ptr->cache                      = new transform::cache();

  // avoid new and delete which requires the allocated and freed types to match
  static_assert(std::is_trivially_copyable_v<decltype(policy_sel)>);
  build_ptr->runtime_policy = std::malloc(sizeof(policy_sel));
  std::memcpy(build_ptr->runtime_policy, &policy_sel, sizeof(policy_sel));

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_binary_transform_build(): %s\n", exc.what());
  fflush(stdout);
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_binary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    error = static_cast<CUresult>(transform::cdt::dispatch<transform::cdt::requires_stable_address::no>(
      ::cuda::std::make_tuple<indirect_iterator_t, indirect_iterator_t>(d_in1, d_in2),
      indirect_iterator_t{d_out},
      static_cast<OffsetT>(num_items),
      transform::cdt::always_true_predicate{},
      op,
      stream,
      *static_cast<cub::detail::transform::policy_selector<2>*>(build.runtime_policy),
      transform::transform_kernel_source<2>{
        build, {transform::make_iterator_info(d_in1), transform::make_iterator_info(d_in2)}},
      cub::detail::CudaDriverLauncherFactory{cu_device, build.cc}));
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_binary_transform(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }
  if (pushed)
  {
    CUcontext cu_context;
    cuCtxPopCurrent(&cu_context);
  }
  return error;
}

CUresult cccl_device_unary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_unary_transform_build_ex(
    build_ptr, d_in, d_out, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}

CUresult cccl_device_binary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_binary_transform_build_ex(
    build_ptr, d_in1, d_in2, d_out, op, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path, nullptr);
}

CUresult cccl_device_transform_cleanup(cccl_device_transform_build_result_t* build_ptr)

try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  using namespace cub::detail::transform;
  std::unique_ptr<char[]> cubin(static_cast<char*>(build_ptr->cubin));
  std::free(build_ptr->runtime_policy);
  std::unique_ptr<transform::cache> cache(static_cast<transform::cache*>(build_ptr->cache));
  check(cuLibraryUnload(build_ptr->library));

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fflush(stderr);
  printf("\nEXCEPTION in cccl_device_transform_cleanup(): %s\n", exc.what());
  fflush(stdout);

  return CUDA_ERROR_UNKNOWN;
}
