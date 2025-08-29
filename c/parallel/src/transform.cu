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
#include <util/runtime_policy.h>
#include <util/types.h>

struct op_wrapper;
struct device_transform_policy;

using OffsetT = long;
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
  check(nvrtcGetTypeName<device_transform_policy>(&chained_policy_t));

  const std::string input_iterator_t  = get_iterator_name<input_storage_t>(input_it, input_iterator_name);
  const std::string output_iterator_t = get_iterator_name<output_storage_t>(output_it, output_iterator_name);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&transform_op_t));

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
  check(nvrtcGetTypeName<device_transform_policy>(&chained_policy_t));

  const std::string input1_iterator_t = get_iterator_name<input1_storage_t>(input1_it, input1_iterator_name);
  const std::string input2_iterator_t = get_iterator_name<input2_storage_t>(input2_it, input2_iterator_name);
  const std::string output_iterator_t = get_iterator_name<output_storage_t>(output_it, output_iterator_name);

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string transform_op_t;
  check(nvrtcGetTypeName<op_wrapper>(&transform_op_t));

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

template <typename AgentPolicy>
struct runtime_tuning_policy_variant
{
  using max_policy = runtime_tuning_policy_variant;

  cdt::Algorithm algorithm;
  int min_bif;
  AgentPolicy algo_policy;

  cdt::Algorithm Algorithm() const
  {
    return algorithm;
  }

  int MinBif() const
  {
    return min_bif;
  }

  AgentPolicy AlgorithmPolicy() const
  {
    return algo_policy;
  }

  template <typename F>
  cudaError_t Invoke([[maybe_unused]] int device_ptx_version, F& op)
  {
    return op.template Invoke<runtime_tuning_policy_variant>(*this);
  }
};

using runtime_tuning_policy =
  std::variant<runtime_tuning_policy_variant<cdt::RuntimeTransformAgentPrefetchPolicy>,
               runtime_tuning_policy_variant<cdt::RuntimeTransformAgentVectorizedPolicy>,
               runtime_tuning_policy_variant<cdt::RuntimeTransformAgentAsyncPolicy>>;

runtime_tuning_policy* make_runtime_tuning_policy(
  cdt::Algorithm algorithm,
  int min_bif,
  std::variant<cdt::RuntimeTransformAgentPrefetchPolicy,
               cdt::RuntimeTransformAgentVectorizedPolicy,
               cdt::RuntimeTransformAgentAsyncPolicy> algo_policy)
{
  return new auto(std::visit(
    [&](auto policy) -> runtime_tuning_policy {
      return runtime_tuning_policy_variant{algorithm, min_bif, policy};
    },
    algo_policy));
}

template <int NumInputs>
struct transform_kernel_source
{
  cccl_device_transform_build_result_t& build;
  std::array<cuda::std::pair<cuda::std::size_t, cuda::std::size_t>, NumInputs> it_value_sizes_alignments;

  static constexpr bool CanCacheConfiguration()
  {
    return false;
  }

  CUkernel TransformKernel() const
  {
    return build.transform_kernel;
  }

  int LoadedBytesPerIteration() const
  {
    return build.loaded_bytes_per_iteration;
  }

  auto ItValueSizesAlignments() const
  {
    return cuda::std::span(it_value_sizes_alignments);
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

  static auto IsPointerAligned(indirect_iterator_t it, int alignment)
  {
    return it.value_size != 0 && ::cuda::is_aligned(*static_cast<char**>(it.ptr), alignment);
  }
};

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
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    const char* name = "test";

    const int cc                 = cc_major * 10 + cc_minor;
    const auto input_it_value_t  = cccl_type_enum_to_name<input_storage_t>(input_it.value_type.type);
    const auto output_it_value_t = cccl_type_enum_to_name<output_storage_t>(output_it.value_type.type);
    const auto offset_t          = cccl_type_enum_to_name(cccl_type_enum::CCCL_INT64);
    const std::string input_iterator_src =
      make_kernel_input_iterator(offset_t, transform::input_iterator_name, input_it_value_t, input_it);
    const std::string output_iterator_src =
      make_kernel_output_iterator(offset_t, transform::output_iterator_name, output_it_value_t, output_it);
    const std::string op_src = make_kernel_user_unary_operator(input_it_value_t, output_it_value_t, op);

    const std::string ptx_arch = std::format("-arch=compute_{}{}", cc_major, cc_minor);

    constexpr size_t ptx_num_args      = 6;
    const char* ptx_args[ptx_num_args] = {
      ptx_arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true"};

    std::string src = std::format(
      R"XXX(
#include <cub/device/dispatch/tuning/tuning_transform.cuh>
struct __align__({1}) input_storage_t {{
  char data[{0}];
}};
struct __align__({3}) output_storage_t {{
  char data[{2}];
}};
{4}
{5}
{6}
)XXX",
      input_it.value_type.size, // 0
      input_it.value_type.alignment, // 1
      output_it.value_type.size, // 2
      output_it.value_type.alignment, // 3
      input_iterator_src, // 4
      output_iterator_src, // 5
      op_src); // 6

    nlohmann::json runtime_policy = get_policy(
      std::format("cub::detail::transform::MakeTransformPolicyWrapper(cub::detail::transform::policy_hub<false, true, "
                  "::cuda::std::tuple<{0}>, {1}>::max_policy::ActivePolicy{{}})",
                  transform::get_iterator_name<input_storage_t>(input_it, transform::input_iterator_name),
                  transform::get_iterator_name<output_storage_t>(output_it, transform::output_iterator_name)),
      "#include <cub/device/dispatch/tuning/tuning_transform.cuh>\n" + src,
      ptx_args);

    auto algorithm = static_cast<transform::cdt::Algorithm>(runtime_policy["algorithm"].get<int>());
    auto min_bif   = static_cast<int>(runtime_policy["min_bif"].get<int>());

    auto [transform_policy, transform_policy_src] =
      [&]() -> std::tuple<std::variant<transform::cdt::RuntimeTransformAgentPrefetchPolicy,
                                       transform::cdt::RuntimeTransformAgentVectorizedPolicy,
                                       transform::cdt::RuntimeTransformAgentAsyncPolicy>,
                          std::string> {
      switch (algorithm)
      {
        case transform::cdt::Algorithm::prefetch:
          return transform::cdt::RuntimeTransformAgentPrefetchPolicy::from_json(runtime_policy, "algo_policy");
        case transform::cdt::Algorithm::vectorized:
          return transform::cdt::RuntimeTransformAgentVectorizedPolicy::from_json(runtime_policy, "algo_policy");
        case transform::cdt::Algorithm::memcpy_async:
          [[fallthrough]];
        case transform::cdt::Algorithm::ublkcp:
          return transform::cdt::RuntimeTransformAgentAsyncPolicy::from_json(runtime_policy, "algo_policy");
      }
      _CCCL_UNREACHABLE();
    }();

    std::string final_src = std::format(
      R"XXX(
#include <cub/device/dispatch/kernels/transform.cuh>
{0}
struct device_transform_policy {{
  struct ActivePolicy {{
    static constexpr auto algorithm = static_cast<cub::detail::transform::Algorithm>({1});
    static constexpr int min_bif = {2};
    {3}
  }};
}};
)XXX",
      src,
      static_cast<int>(algorithm),
      min_bif,
      transform_policy_src);

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
      "-DCUB_DISABLE_CDP"};

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

    build_ptr->loaded_bytes_per_iteration = input_it.value_type.size;
    build_ptr->cc                         = cc;
    build_ptr->cubin                      = (void*) result.data.release();
    build_ptr->cubin_size                 = result.size;
    build_ptr->runtime_policy             = transform::make_runtime_tuning_policy(algorithm, min_bif, transform_policy);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_unary_transform_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
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
    error = static_cast<CUresult>(std::visit(
      [&]<typename Policy>(Policy policy) {
        return transform::cdt::dispatch_t<
          transform::cdt::requires_stable_address::no, // TODO implement yes
          OffsetT,
          ::cuda::std::tuple<indirect_iterator_t>,
          indirect_iterator_t,
          transform::cdt::always_true_predicate,
          indirect_arg_t,
          Policy,
          transform::transform_kernel_source<1>,
          cub::detail::CudaDriverLauncherFactory>::
          dispatch(d_in,
                   d_out,
                   num_items,
                   {},
                   op,
                   stream,
                   {build, {{{d_in.value_type.size, d_in.value_type.alignment}}}},
                   cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
                   policy);
      },
      *reinterpret_cast<transform::runtime_tuning_policy*>(build.runtime_policy)));
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
{
  CUresult error = CUDA_SUCCESS;

  try
  {
    const char* name = "test";

    const int cc                 = cc_major * 10 + cc_minor;
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

    const std::string ptx_arch = std::format("-arch=compute_{}{}", cc_major, cc_minor);

    constexpr size_t ptx_num_args      = 6;
    const char* ptx_args[ptx_num_args] = {
      ptx_arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true"};

    std::string src = std::format(
      R"XXX(
#include <cub/device/dispatch/kernels/transform.cuh>
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
      op_src); // 9

    nlohmann::json runtime_policy = get_policy(
      std::format("cub::detail::transform::MakeTransformPolicyWrapper(cub::detail::transform::policy_hub<false, true, "
                  "::cuda::std::tuple<{0}, {1}>, {2}>::max_policy::ActivePolicy{{}})",
                  transform::get_iterator_name<input1_storage_t>(input1_it, transform::input1_iterator_name),
                  transform::get_iterator_name<input2_storage_t>(input2_it, transform::input2_iterator_name),
                  transform::get_iterator_name<output_storage_t>(output_it, transform::output_iterator_name)),
      "#include <cub/device/dispatch/tuning/tuning_transform.cuh>\n" + src,
      ptx_args);

    auto algorithm = static_cast<transform::cdt::Algorithm>(runtime_policy["algorithm"].get<int>());
    auto min_bif   = static_cast<int>(runtime_policy["min_bif"].get<int>());

    auto [transform_policy, transform_policy_src] =
      [&]() -> std::tuple<std::variant<transform::cdt::RuntimeTransformAgentPrefetchPolicy,
                                       transform::cdt::RuntimeTransformAgentVectorizedPolicy,
                                       transform::cdt::RuntimeTransformAgentAsyncPolicy>,
                          std::string> {
      switch (algorithm)
      {
        case transform::cdt::Algorithm::prefetch:
          return transform::cdt::RuntimeTransformAgentPrefetchPolicy::from_json(runtime_policy, "algo_policy");
        case transform::cdt::Algorithm::vectorized:
          return transform::cdt::RuntimeTransformAgentVectorizedPolicy::from_json(runtime_policy, "algo_policy");
        case transform::cdt::Algorithm::memcpy_async:
          [[fallthrough]];
        case transform::cdt::Algorithm::ublkcp:
          return transform::cdt::RuntimeTransformAgentAsyncPolicy::from_json(runtime_policy, "algo_policy");
      }
      _CCCL_UNREACHABLE();
    }();

    std::string final_src = std::format(
      R"XXX(
#include <cub/device/dispatch/kernels/transform.cuh>
{0}
struct device_transform_policy {{
  struct ActivePolicy {{
    static constexpr auto algorithm = static_cast<cub::detail::transform::Algorithm>({1});
    static constexpr int min_bif = {2};
    {3}
  }};
}};
)XXX",
      src,
      static_cast<int>(algorithm),
      min_bif,
      transform_policy_src);

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
      "-DCUB_DISABLE_CDP"};

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

    build_ptr->loaded_bytes_per_iteration = (input1_it.value_type.size + input2_it.value_type.size);
    build_ptr->cc                         = cc;
    build_ptr->cubin                      = (void*) result.data.release();
    build_ptr->cubin_size                 = result.size;
    build_ptr->runtime_policy             = transform::make_runtime_tuning_policy(algorithm, min_bif, transform_policy);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_binary_transform_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
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

    error = static_cast<CUresult>(std::visit(
      [&]<typename Policy>(Policy policy) {
        return transform::cdt::dispatch_t<
          transform::cdt::requires_stable_address::no, // TODO implement yes
          OffsetT,
          ::cuda::std::tuple<indirect_iterator_t, indirect_iterator_t>,
          indirect_iterator_t,
          transform::cdt::always_true_predicate,
          indirect_arg_t,
          Policy,
          transform::transform_kernel_source<2>,
          cub::detail::CudaDriverLauncherFactory>::
          dispatch(
            ::cuda::std::make_tuple<indirect_iterator_t, indirect_iterator_t>(d_in1, d_in2),
            d_out,
            num_items,
            {},
            op,
            stream,
            {build,
             {{{d_in1.value_type.size, d_in1.value_type.alignment},
               {d_in2.value_type.size, d_in2.value_type.alignment}}}},
            cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
            policy);
      },
      *reinterpret_cast<transform::runtime_tuning_policy*>(build.runtime_policy)));
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
{
  try
  {
    if (build_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
    std::unique_ptr<transform::runtime_tuning_policy> rtp(
      reinterpret_cast<transform::runtime_tuning_policy*>(build_ptr->runtime_policy));
    check(cuLibraryUnload(build_ptr->library));
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_transform_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
