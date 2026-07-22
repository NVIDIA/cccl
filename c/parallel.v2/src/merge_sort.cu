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

#include <cccl/c/merge_sort.h>
#include <hostjit/codegen/cub_call.hpp>
#include <util/build_utils.h>
#include <util/serialization.h>

using namespace hostjit::codegen;

// Keys-only: (temp, temp_bytes, in_keys, out_keys, num_items, cmp_state, stream)
using keys_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*);
// Key-value pairs: (temp, temp_bytes, in_keys, in_items, out_keys, out_items, num_items, cmp_state, stream)
using pairs_fn_t = int (*)(void*, size_t*, void*, void*, void*, void*, unsigned long long, void*, void*);

static bool is_null_items(cccl_iterator_t it)
{
  return it.type == CCCL_POINTER && it.state == nullptr;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_merge_sort_build_ex(
  cccl_device_merge_sort_build_result_t* build_ptr,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
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
  if (d_out_keys.type == CCCL_ITERATOR || d_out_items.type == CCCL_ITERATOR)
  {
    fprintf(stderr, "\nERROR in cccl_device_merge_sort_build(): merge sort output cannot be an iterator\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  const bool has_items = !is_null_items(d_in_items);

  CubCallResult result = [&] {
    if (has_items)
    {
      return CubCall::from("cub/device/device_merge_sort.cuh")
        .run("cub::DeviceMergeSort::SortPairsCopy")
        .name("cccl_jit_merge_sort")
        .with(temp_storage,
              temp_bytes,
              in(d_in_keys),
              in(d_in_items),
              out(d_out_keys),
              out(d_out_items),
              num_items,
              cmp(op),
              stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
    else
    {
      return CubCall::from("cub/device/device_merge_sort.cuh")
        .run("cub::DeviceMergeSort::SortKeysCopy")
        .name("cccl_jit_merge_sort")
        .with(temp_storage, temp_bytes, in(d_in_keys), out(d_out_keys), num_items, cmp(op), stream)
        .compile(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
  }();

  build_ptr->cc = cc_major * 10 + cc_minor;
  // Populate payload with the compiled artifact bytes too (not just the
  // live loaded state) so cccl_device_merge_sort_serialize works uniformly
  // whether the build_result came from this fused path or from
  // _compile()+_load(). Best-effort: a failed read leaves payload null,
  // which only affects later serialize() calls, not this build itself.
  auto library_bytes = cccl::detail::read_compiled_library_bytes(result.compiler);
  cccl::detail::copy_bytes(library_bytes, build_ptr->payload, build_ptr->payload_size);
  build_ptr->jit_compiler = result.compiler;
  build_ptr->sort_fn      = result.fn_ptr;
  build_ptr->keys_only    = has_items ? 0 : 1;
  build_ptr->key_type     = d_in_keys.value_type;
  build_ptr->item_type    = d_in_items.value_type;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_merge_sort_compile(
  cccl_device_merge_sort_build_result_t* build_ptr,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
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
  if (d_out_keys.type == CCCL_ITERATOR || d_out_items.type == CCCL_ITERATOR)
  {
    fprintf(stderr, "\nERROR in cccl_device_merge_sort_compile(): merge sort output cannot be an iterator\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();
  cccl::detail::MergedBuildConfig merged(config, cub_path, thrust_path);

  const bool has_items = !is_null_items(d_in_items);

  CubCallCompileOnlyResult result = [&] {
    if (has_items)
    {
      return CubCall::from("cub/device/device_merge_sort.cuh")
        .run("cub::DeviceMergeSort::SortPairsCopy")
        .name("cccl_jit_merge_sort")
        .with(temp_storage,
              temp_bytes,
              in(d_in_keys),
              in(d_in_items),
              out(d_out_keys),
              out(d_out_items),
              num_items,
              cmp(op),
              stream)
        .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
    else
    {
      return CubCall::from("cub/device/device_merge_sort.cuh")
        .run("cub::DeviceMergeSort::SortKeysCopy")
        .name("cccl_jit_merge_sort")
        .with(temp_storage, temp_bytes, in(d_in_keys), out(d_out_keys), num_items, cmp(op), stream)
        .compileOnly(cc_major, cc_minor, merged.get(), ctk_root, cccl_include_path);
    }
  }();

  build_ptr->cc = cc_major * 10 + cc_minor;
  cccl::detail::copy_bytes(result.library_bytes, build_ptr->payload, build_ptr->payload_size);
  // Zero-init fields set by _load, not _compile (matches v1's contract).
  build_ptr->jit_compiler = nullptr;
  build_ptr->sort_fn      = nullptr;
  build_ptr->keys_only    = has_items ? 0 : 1;
  build_ptr->key_type     = d_in_keys.value_type;
  build_ptr->item_type    = d_in_items.value_type;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort_compile(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_merge_sort_load(cccl_device_merge_sort_build_result_t* build_ptr, const char* ctk_path)
try
{
  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  hostjit::CompilerConfig jit_config = cccl::detail::make_load_jit_config(ctk_path);

  auto compiler = std::make_unique<hostjit::JITCompiler>(jit_config);
  std::vector<char> library_bytes(
    static_cast<char*>(build_ptr->payload), static_cast<char*>(build_ptr->payload) + build_ptr->payload_size);
  if (!compiler->loadFromBytes(library_bytes))
  {
    fprintf(stderr, "\nERROR in cccl_device_merge_sort_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  // Both keys-only and key-value builds export the same symbol name
  // ("cccl_jit_merge_sort"); the caller (cccl_device_merge_sort) dispatches
  // to the correct function-pointer type via build.keys_only.
  using generic_fn_t = void*;
  auto fn            = compiler->getFunction<generic_fn_t>("cccl_jit_merge_sort");
  if (!fn)
  {
    fprintf(stderr, "\nERROR in cccl_device_merge_sort_load(): %s\n", compiler->getLastError().c_str());
    return CUDA_ERROR_UNKNOWN;
  }

  build_ptr->jit_compiler = compiler.release();
  build_ptr->sort_fn      = fn;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort_load(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_merge_sort_serialize(
  const cccl_device_merge_sort_build_result_t* build_ptr, void** out_buf, size_t* out_size)
try
{
  *out_buf  = nullptr;
  *out_size = 0;

  if (build_ptr == nullptr || build_ptr->payload == nullptr || build_ptr->payload_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // extra = {keys_only (u32), key_type (size,alignment,type: 2xu64+u32), item_type (2xu64+u32)}
  std::vector<char> extra;
  auto append_type = [&](cccl_type_info t) {
    uint64_t size = t.size, alignment = t.alignment;
    uint32_t type = static_cast<uint32_t>(t.type);
    extra.insert(extra.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
    extra.insert(
      extra.end(), reinterpret_cast<char*>(&alignment), reinterpret_cast<char*>(&alignment) + sizeof(alignment));
    extra.insert(extra.end(), reinterpret_cast<char*>(&type), reinterpret_cast<char*>(&type) + sizeof(type));
  };
  uint32_t keys_only_u32 = static_cast<uint32_t>(build_ptr->keys_only);
  extra.insert(extra.end(),
               reinterpret_cast<char*>(&keys_only_u32),
               reinterpret_cast<char*>(&keys_only_u32) + sizeof(keys_only_u32));
  append_type(build_ptr->key_type);
  append_type(build_ptr->item_type);

  std::vector<char> blob = cccl::serialization_v2::write_blob(
    CCCL_SERIALIZATION_V2_ALGO_MERGE_SORT,
    build_ptr->cc,
    {"cccl_jit_merge_sort"},
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
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort_serialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult
cccl_device_merge_sort_deserialize(cccl_device_merge_sort_build_result_t* build_ptr, const void* buf, size_t size)
try
{
  if (build_ptr == nullptr || buf == nullptr || size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::serialization_v2::parsed_blob parsed =
    cccl::serialization_v2::read_blob(CCCL_SERIALIZATION_V2_ALGO_MERGE_SORT, buf, size);

  constexpr size_t type_sz  = sizeof(uint64_t) * 2 + sizeof(uint32_t);
  constexpr size_t extra_sz = sizeof(uint32_t) + type_sz * 2;
  if (parsed.extra.size() != extra_sz || parsed.symbol_names.size() != 1)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  size_t pos = 0;
  uint32_t keys_only_u32;
  std::memcpy(&keys_only_u32, parsed.extra.data() + pos, sizeof(keys_only_u32));
  pos += sizeof(keys_only_u32);

  auto read_type = [&](cccl_type_info& out) {
    uint64_t type_size, alignment;
    uint32_t type;
    std::memcpy(&type_size, parsed.extra.data() + pos, sizeof(type_size));
    pos += sizeof(type_size);
    std::memcpy(&alignment, parsed.extra.data() + pos, sizeof(alignment));
    pos += sizeof(alignment);
    std::memcpy(&type, parsed.extra.data() + pos, sizeof(type));
    pos += sizeof(type);
    out =
      cccl_type_info{static_cast<size_t>(type_size), static_cast<size_t>(alignment), static_cast<cccl_type_enum>(type)};
  };

  // Commit-on-success: build a local result{} and only assign to
  // *build_ptr after every read succeeds, so *build_ptr is left unchanged
  // on failure (matches v1's deserialize contract).
  cccl_device_merge_sort_build_result_t result{};
  read_type(result.key_type);
  read_type(result.item_type);

  result.cc = parsed.cc;
  cccl::detail::copy_bytes(
    std::vector<char>(parsed.payload, parsed.payload + parsed.payload_size), result.payload, result.payload_size);
  result.jit_compiler = nullptr;
  result.sort_fn      = nullptr;
  result.keys_only    = static_cast<int>(keys_only_u32);

  *build_ptr = result;
  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort_deserialize(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_merge_sort_build(
  cccl_device_merge_sort_build_result_t* build,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_merge_sort_build_ex(
    build,
    d_in_keys,
    d_in_items,
    d_out_keys,
    d_out_items,
    op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

CUresult cccl_device_merge_sort(
  cccl_device_merge_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
try
{
  if (!build.sort_fn)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  int status;
  // Dispatch to the correct function arity. build.keys_only was set at
  // build time, so we don't have to re-derive the pairs-vs-keys decision
  // from the iterator arguments here (and they must match the build for
  // the function-pointer types to be valid).
  if (!build.keys_only)
  {
    // Pairs build: (temp, temp_bytes, in_keys, in_items, out_keys, out_items, num_items, cmp_state, stream)
    auto fn = reinterpret_cast<pairs_fn_t>(build.sort_fn);
    status  = fn(
      d_temp_storage,
      temp_storage_bytes,
      d_in_keys.state,
      d_in_items.state,
      d_out_keys.state,
      d_out_items.state,
      num_items,
      op.state,
      reinterpret_cast<void*>(stream));
  }
  else
  {
    // Keys-only build: (temp, temp_bytes, in_keys, out_keys, num_items, cmp_state, stream)
    auto fn = reinterpret_cast<keys_fn_t>(build.sort_fn);
    status =
      fn(d_temp_storage,
         temp_storage_bytes,
         d_in_keys.state,
         d_out_keys.state,
         num_items,
         op.state,
         reinterpret_cast<void*>(stream));
  }

  return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_merge_sort_cleanup(cccl_device_merge_sort_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  cccl::detail::release_jit_artifacts(build_ptr);
  build_ptr->sort_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
