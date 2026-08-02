//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <cccl/c/radix_sort.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>
#include <util/serialization.h>

using namespace hostjit::codegen;

static bool is_null_it(cccl_iterator_t it)
{
  return it.type == CCCL_POINTER && it.state == nullptr;
}

static bool is_null_op(cccl_op_t op)
{
  return op.name == nullptr || op.name[0] == '\0';
}

// Two JIT wrappers are produced by CubCall per build:
//
//   COPY variant — wraps cub::DeviceRadixSort::Sort{Keys,Pairs}{,Descending}'s
//   copy-overload. Result is always in *_out; selector is implicitly 0. Used
//   when the caller invokes the run-time API with is_overwrite_okay=false.
//     keys-only: fn(temp, temp_bytes, keys_in, keys_out, num_items,
//                   &begin_bit, &end_bit, stream)
//     pairs:     fn(temp, temp_bytes, keys_in, keys_out, values_in, values_out,
//                   num_items, &begin_bit, &end_bit, stream)
//
//   DOUBLE-BUFFER (overwrite) variant — wraps the DoubleBuffer overload.
//   Constructs cub::DoubleBuffer<KeyT>(keys_in, keys_out) (and ValueT for
//   pairs), runs the sort, then writes the buffer's `selector` (0 or 1) to a
//   host-provided int*. Result may live in either keys_in or keys_out depending
//   on the number of CUB passes — the selector tells the caller which.
//     keys-only: fn(temp, temp_bytes, keys_in, keys_out, num_items,
//                   &begin_bit, &end_bit, selector_out, stream)
//     pairs:     fn(temp, temp_bytes, keys_in, keys_out, values_in, values_out,
//                   num_items, &begin_bit, &end_bit, selector_out, stream)
//
// begin_bit/end_bit go through CubCall::typed_scalar (host-pointer + memcpy
// onto the stack inside the JIT wrapper).
//
// Decomposer: only identity (null decomposer) is supported.
using radix_sort_keys_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*, void*);
using radix_sort_pairs_fn_t =
  int (*)(void*, size_t*, void*, void*, void*, void*, unsigned long long, void*, void*, void*);
using radix_sort_keys_overwrite_fn_t =
  int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*, void*, void*);
using radix_sort_pairs_overwrite_fn_t =
  int (*)(void*, size_t*, void*, void*, void*, void*, unsigned long long, void*, void*, void*, void*);

// Type info for the begin_bit/end_bit int scalars passed to CubCall.
static constexpr cccl_type_info k_int_type{sizeof(int), alignof(int), CCCL_INT32};

CUresult cccl_device_radix_sort_build_ex(
  cccl_device_radix_sort_build_result_t* build_ptr,
  cccl_sort_order_t sort_order,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_op_t decomposer,
  const char* /*decomposer_return_type*/,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  if (!is_null_op(decomposer))
  {
    fprintf(stderr,
            "\nERROR in cccl_device_radix_sort_build(): custom radix decomposers are not supported "
            "in the HostJIT path. Use standard integer/float key types.\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  const bool keys_only = is_null_it(input_values_it);
  const bool ascending = (sort_order == CCCL_ASCENDING);

  // CUB writes to caller-provided device pointers at run time. Build needs
  // an iterator descriptor for the outputs; synthesize raw-pointer ones with
  // the same value_type as the matching input.
  cccl_iterator_t output_keys_it = input_keys_it;
  output_keys_it.type            = CCCL_POINTER;
  output_keys_it.state           = nullptr;
  cccl_iterator_t output_values_it{};
  output_values_it.type       = CCCL_POINTER;
  output_values_it.state      = nullptr;
  output_values_it.value_type = input_values_it.value_type;

  const char* cub_algo;
  if (keys_only)
  {
    cub_algo = ascending ? "cub::DeviceRadixSort::SortKeys" : "cub::DeviceRadixSort::SortKeysDescending";
  }
  else
  {
    cub_algo = ascending ? "cub::DeviceRadixSort::SortPairs" : "cub::DeviceRadixSort::SortPairsDescending";
  }

  auto cb_copy = [&] {
    if (keys_only)
    {
      return CubCall::from("cub/device/device_radix_sort.cuh")
        .run(cub_algo)
        .name("cccl_jit_radix_sort")
        .with(temp_storage,
              temp_bytes,
              in(input_keys_it),
              out(output_keys_it),
              num_items,
              typed_scalar(k_int_type, "begin_bit"),
              typed_scalar(k_int_type, "end_bit"),
              stream);
    }
    return CubCall::from("cub/device/device_radix_sort.cuh")
      .run(cub_algo)
      .name("cccl_jit_radix_sort")
      .with(temp_storage,
            temp_bytes,
            in(input_keys_it),
            out(output_keys_it),
            in(input_values_it),
            out(output_values_it),
            num_items,
            typed_scalar(k_int_type, "begin_bit"),
            typed_scalar(k_int_type, "end_bit"),
            stream);
  }();

  auto cb_overwrite = [&] {
    if (keys_only)
    {
      return CubCall::from("cub/device/device_radix_sort.cuh")
        .run(cub_algo)
        .name("cccl_jit_radix_sort_overwrite")
        .with(temp_storage,
              temp_bytes,
              double_buffer(input_keys_it, output_keys_it, "d_keys_buffer"),
              num_items,
              typed_scalar(k_int_type, "begin_bit"),
              typed_scalar(k_int_type, "end_bit"),
              selector_out("d_keys_buffer"),
              stream);
    }
    return CubCall::from("cub/device/device_radix_sort.cuh")
      .run(cub_algo)
      .name("cccl_jit_radix_sort_overwrite")
      .with(temp_storage,
            temp_bytes,
            double_buffer(input_keys_it, output_keys_it, "d_keys_buffer"),
            double_buffer(input_values_it, output_values_it, "d_values_buffer"),
            num_items,
            typed_scalar(k_int_type, "begin_bit"),
            typed_scalar(k_int_type, "end_bit"),
            selector_out("d_keys_buffer"),
            stream);
  }();

  auto result =
    CubCall::compile({cb_copy, cb_overwrite}, cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build_ptr->cc      = cc_major * 10 + cc_minor;
  auto library_bytes = cccl::detail::read_compiled_library_bytes(result.compiler);
  cccl::detail::copy_bytes(library_bytes, build_ptr->payload, build_ptr->payload_size);
  build_ptr->jit_compiler      = result.compiler;
  build_ptr->sort_fn           = result.fn_ptrs[0];
  build_ptr->sort_fn_overwrite = result.fn_ptrs[1];
  build_ptr->key_type          = input_keys_it.value_type;
  build_ptr->value_type        = input_values_it.value_type;
  build_ptr->order             = sort_order;
  build_ptr->keys_only         = keys_only ? 1 : 0;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_radix_sort_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_radix_sort_compile(
  cccl_device_radix_sort_build_result_t* build_ptr,
  cccl_sort_order_t sort_order,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_op_t decomposer,
  const char* /*decomposer_return_type*/,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
try
{
  if (!is_null_op(decomposer))
  {
    fprintf(stderr,
            "\nERROR in cccl_device_radix_sort_compile(): custom radix decomposers are not supported "
            "in the HostJIT path. Use standard integer/float key types.\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  const bool keys_only = is_null_it(input_values_it);
  const bool ascending = (sort_order == CCCL_ASCENDING);

  cccl_iterator_t output_keys_it = input_keys_it;
  output_keys_it.type            = CCCL_POINTER;
  output_keys_it.state           = nullptr;
  cccl_iterator_t output_values_it{};
  output_values_it.type       = CCCL_POINTER;
  output_values_it.state      = nullptr;
  output_values_it.value_type = input_values_it.value_type;

  const char* cub_algo;
  if (keys_only)
  {
    cub_algo = ascending ? "cub::DeviceRadixSort::SortKeys" : "cub::DeviceRadixSort::SortKeysDescending";
  }
  else
  {
    cub_algo = ascending ? "cub::DeviceRadixSort::SortPairs" : "cub::DeviceRadixSort::SortPairsDescending";
  }

  auto cb_copy = [&] {
    if (keys_only)
    {
      return CubCall::from("cub/device/device_radix_sort.cuh")
        .run(cub_algo)
        .name("cccl_jit_radix_sort")
        .with(temp_storage,
              temp_bytes,
              in(input_keys_it),
              out(output_keys_it),
              num_items,
              typed_scalar(k_int_type, "begin_bit"),
              typed_scalar(k_int_type, "end_bit"),
              stream);
    }
    return CubCall::from("cub/device/device_radix_sort.cuh")
      .run(cub_algo)
      .name("cccl_jit_radix_sort")
      .with(temp_storage,
            temp_bytes,
            in(input_keys_it),
            out(output_keys_it),
            in(input_values_it),
            out(output_values_it),
            num_items,
            typed_scalar(k_int_type, "begin_bit"),
            typed_scalar(k_int_type, "end_bit"),
            stream);
  }();

  auto cb_overwrite = [&] {
    if (keys_only)
    {
      return CubCall::from("cub/device/device_radix_sort.cuh")
        .run(cub_algo)
        .name("cccl_jit_radix_sort_overwrite")
        .with(temp_storage,
              temp_bytes,
              double_buffer(input_keys_it, output_keys_it, "d_keys_buffer"),
              num_items,
              typed_scalar(k_int_type, "begin_bit"),
              typed_scalar(k_int_type, "end_bit"),
              selector_out("d_keys_buffer"),
              stream);
    }
    return CubCall::from("cub/device/device_radix_sort.cuh")
      .run(cub_algo)
      .name("cccl_jit_radix_sort_overwrite")
      .with(temp_storage,
            temp_bytes,
            double_buffer(input_keys_it, output_keys_it, "d_keys_buffer"),
            double_buffer(input_values_it, output_values_it, "d_values_buffer"),
            num_items,
            typed_scalar(k_int_type, "begin_bit"),
            typed_scalar(k_int_type, "end_bit"),
            selector_out("d_keys_buffer"),
            stream);
  }();

  auto result =
    CubCall::compileOnly({cb_copy, cb_overwrite}, cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);

  build_ptr->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_bytes(result.library_bytes, build_ptr->payload, build_ptr->payload_size);
  build_ptr->jit_compiler      = nullptr;
  build_ptr->sort_fn           = nullptr;
  build_ptr->sort_fn_overwrite = nullptr;
  build_ptr->key_type          = input_keys_it.value_type;
  build_ptr->value_type        = input_values_it.value_type;
  build_ptr->order             = sort_order;
  build_ptr->keys_only         = keys_only ? 1 : 0;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_radix_sort_compile(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_radix_sort_load(cccl_device_radix_sort_build_result_t* build_ptr, const char* ctk_path)
try
{
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  hostjit::CompilerConfig jit_config = cccl::detail::make_load_jit_config(ctk_path);
  auto compiler                      = std::make_unique<hostjit::JITCompiler>(jit_config);
  std::vector<char> library_bytes(
    static_cast<char*>(build_ptr->payload), static_cast<char*>(build_ptr->payload) + build_ptr->payload_size);
  if (!compiler->loadFromBytes(library_bytes))
  {
    fprintf(stderr, "\nERROR in cccl_device_radix_sort_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }
  using generic_fn_t = void*;
  auto fn            = compiler->getFunction<generic_fn_t>("cccl_jit_radix_sort");
  auto fn_overwrite  = compiler->getFunction<generic_fn_t>("cccl_jit_radix_sort_overwrite");
  if (!fn || !fn_overwrite)
  {
    fprintf(stderr, "\nERROR in cccl_device_radix_sort_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  build_ptr->jit_compiler      = compiler.release();
  build_ptr->sort_fn           = fn;
  build_ptr->sort_fn_overwrite = fn_overwrite;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_radix_sort_load(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_radix_sort_serialize(
  const cccl_device_radix_sort_build_result_t* build_ptr, void** out_buf, size_t* out_size)
try
{
  *out_buf  = nullptr;
  *out_size = 0;
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // extra = {key_type (size,alignment,type: 3xu64), value_type (3xu64), order (u32), keys_only (u32)}
  std::vector<char> extra;
  auto append_type = [&](cccl_type_info t) {
    uint64_t size = t.size, alignment = t.alignment;
    uint32_t type = static_cast<uint32_t>(t.type);
    extra.insert(extra.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
    extra.insert(
      extra.end(), reinterpret_cast<char*>(&alignment), reinterpret_cast<char*>(&alignment) + sizeof(alignment));
    extra.insert(extra.end(), reinterpret_cast<char*>(&type), reinterpret_cast<char*>(&type) + sizeof(type));
  };
  append_type(build_ptr->key_type);
  append_type(build_ptr->value_type);
  uint32_t order_u32     = static_cast<uint32_t>(build_ptr->order);
  uint32_t keys_only_u32 = static_cast<uint32_t>(build_ptr->keys_only);
  extra.insert(
    extra.end(), reinterpret_cast<char*>(&order_u32), reinterpret_cast<char*>(&order_u32) + sizeof(order_u32));
  extra.insert(extra.end(),
               reinterpret_cast<char*>(&keys_only_u32),
               reinterpret_cast<char*>(&keys_only_u32) + sizeof(keys_only_u32));

  std::vector<char> blob = cccl::serialization_v2::write_blob(
    CCCL_SERIALIZATION_V2_ALGO_RADIX_SORT,
    build_ptr->cc,
    {"cccl_jit_radix_sort", "cccl_jit_radix_sort_overwrite"},
    extra,
    build_ptr->payload,
    build_ptr->payload_size);
  auto* buf = new char[blob.size()];
  std::memcpy(buf, blob.data(), blob.size());
  *out_buf  = buf;
  *out_size = blob.size();
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_radix_sort_serialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult
cccl_device_radix_sort_deserialize(cccl_device_radix_sort_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cccl::serialization_v2::parsed_blob parsed =
    cccl::serialization_v2::read_blob(CCCL_SERIALIZATION_V2_ALGO_RADIX_SORT, buf, size);
  constexpr size_t type_sz  = sizeof(uint64_t) * 2 + sizeof(uint32_t);
  constexpr size_t extra_sz = type_sz * 2 + sizeof(uint32_t) * 2;
  if (parsed.extra.size() != extra_sz || parsed.symbol_names.size() != 2)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  size_t pos     = 0;
  auto read_type = [&](cccl_type_info& out) {
    uint64_t size, alignment;
    uint32_t type;
    std::memcpy(&size, parsed.extra.data() + pos, sizeof(size));
    pos += sizeof(size);
    std::memcpy(&alignment, parsed.extra.data() + pos, sizeof(alignment));
    pos += sizeof(alignment);
    std::memcpy(&type, parsed.extra.data() + pos, sizeof(type));
    pos += sizeof(type);
    out = cccl_type_info{static_cast<size_t>(size), static_cast<size_t>(alignment), static_cast<cccl_type_enum>(type)};
  };

  cccl_device_radix_sort_build_result_t result{};
  read_type(result.key_type);
  read_type(result.value_type);
  uint32_t order_u32, keys_only_u32;
  std::memcpy(&order_u32, parsed.extra.data() + pos, sizeof(order_u32));
  pos += sizeof(order_u32);
  std::memcpy(&keys_only_u32, parsed.extra.data() + pos, sizeof(keys_only_u32));

  result.cc = parsed.cc;
  cccl::detail::copy_bytes(
    std::vector<char>(parsed.payload, parsed.payload + parsed.payload_size), result.payload, result.payload_size);
  result.jit_compiler      = nullptr;
  result.sort_fn           = nullptr;
  result.sort_fn_overwrite = nullptr;
  result.order             = static_cast<cccl_sort_order_t>(order_u32);
  result.keys_only         = static_cast<int>(keys_only_u32);

  *build_ptr = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_radix_sort_deserialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_radix_sort_build(
  cccl_device_radix_sort_build_result_t* build,
  cccl_sort_order_t sort_order,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_op_t decomposer,
  const char* decomposer_return_type,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_radix_sort_build_ex(
    build,
    sort_order,
    input_keys_it,
    input_values_it,
    decomposer,
    decomposer_return_type,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_radix_sort(
  cccl_device_radix_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t /*decomposer*/,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  bool is_overwrite_okay,
  int* selector,
  CUstream stream)
try
{
  // Dispatch on is_overwrite_okay: the copy variant always lands the result
  // in d_keys_out (selector = 0); the DoubleBuffer variant may land it in
  // either buffer and reports which via its `selector` member, captured here
  // by passing a pointer for the wrapper to write into.
  int status;
  int local_selector = 0;
  if (is_overwrite_okay)
  {
    if (!build.sort_fn_overwrite)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (build.keys_only)
    {
      auto fn = reinterpret_cast<radix_sort_keys_overwrite_fn_t>(build.sort_fn_overwrite);
      status  = fn(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.state,
        d_keys_out.state,
        static_cast<unsigned long long>(num_items),
        &begin_bit,
        &end_bit,
        &local_selector,
        reinterpret_cast<void*>(stream));
    }
    else
    {
      auto fn = reinterpret_cast<radix_sort_pairs_overwrite_fn_t>(build.sort_fn_overwrite);
      status  = fn(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.state,
        d_keys_out.state,
        d_values_in.state,
        d_values_out.state,
        static_cast<unsigned long long>(num_items),
        &begin_bit,
        &end_bit,
        &local_selector,
        reinterpret_cast<void*>(stream));
    }
  }
  else
  {
    if (!build.sort_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (build.keys_only)
    {
      auto fn = reinterpret_cast<radix_sort_keys_fn_t>(build.sort_fn);
      status  = fn(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.state,
        d_keys_out.state,
        static_cast<unsigned long long>(num_items),
        &begin_bit,
        &end_bit,
        reinterpret_cast<void*>(stream));
    }
    else
    {
      auto fn = reinterpret_cast<radix_sort_pairs_fn_t>(build.sort_fn);
      status  = fn(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in.state,
        d_keys_out.state,
        d_values_in.state,
        d_values_out.state,
        static_cast<unsigned long long>(num_items),
        &begin_bit,
        &end_bit,
        reinterpret_cast<void*>(stream));
    }
    local_selector = 0;
  }

  if (selector)
  {
    *selector = local_selector;
  }

  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_radix_sort(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_radix_sort_cleanup(cccl_device_radix_sort_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->sort_fn           = nullptr;
  build_ptr->sort_fn_overwrite = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_radix_sort_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
