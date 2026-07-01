//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cccl/c/aot.h>
#include <cccl/c/types.h>

namespace cccl::aot
{
// 8-byte magic identifying a CCCL AoT blob. Bumping the trailing digit
// signals a hard format break (older blobs become unloadable).
inline constexpr char k_blob_magic[8] = {'C', 'C', 'C', 'L', 'A', 'O', 'T', '1'};

// Per-algorithm format version. Bump when the layout *for that algorithm*
// changes (e.g. a new field added to its build_result_t).
inline constexpr uint32_t k_format_version = 1;

// Fixed-layout header at the start of every blob. Packed POD; layout is
// part of the on-disk format, do not reorder.
struct blob_header
{
  char magic[8]; // k_blob_magic
  uint32_t algo_tag; // cccl_aot_algo_t
  uint32_t format_version; // k_format_version at write time
  uint64_t cccl_version; // CCCL_C_PARALLEL_VERSION at serialize time; mismatched → reject
  uint32_t payload_kind; // cccl_payload_kind_t
  uint32_t cc; // cc_major*10 + cc_minor
};
static_assert(sizeof(blob_header) == 32, "blob_header layout must be stable");

// Append-only byte buffer used by *_serialize implementations.
// Owns a std::vector<char> internally; release() hands back a heap buffer
// allocated with new[] (matching cccl_aot_buffer_free, which does delete[]).
class buffer_writer
{
  std::vector<char> data;

public:
  void write_bytes(const void* p, size_t n)
  {
    if (n == 0)
    {
      return;
    }
    const char* src = static_cast<const char*>(p);
    data.insert(data.end(), src, src + n);
  }

  template <class T>
  void write_pod(const T& v)
  {
    static_assert(std::is_trivially_copyable_v<T>, "write_pod requires trivially-copyable type");
    write_bytes(&v, sizeof(T));
  }

  // Writes a length-prefixed string. Length is uint64_t. nullptr writes as length=0.
  void write_cstring(const char* s)
  {
    const uint64_t n = (s == nullptr) ? 0 : std::strlen(s);
    write_pod<uint64_t>(n);
    if (n > 0)
    {
      write_bytes(s, n);
    }
  }

  // Writes a length-prefixed byte blob.
  void write_blob(const void* p, size_t n)
  {
    write_pod<uint64_t>(n);
    if (n > 0)
    {
      write_bytes(p, n);
    }
  }

  size_t size() const noexcept
  {
    return data.size();
  }

  // Hands back ownership as a new[]'d buffer. After release() the writer is empty.
  void release(void** out_buf, size_t* out_size)
  {
    const size_t n = data.size();
    auto p         = std::make_unique<char[]>(n);
    if (n > 0)
    {
      std::memcpy(p.get(), data.data(), n);
    }
    data.clear();
    *out_buf  = p.release();
    *out_size = n;
  }
};

// Bounds-checked byte buffer reader used by *_deserialize implementations.
// Borrows the input buffer; allocations it produces (via read_cstring_dup,
// read_blob_new) are owned by the caller.
class buffer_reader
{
  const char* pos;
  size_t nrem;

public:
  buffer_reader(const void* buf, size_t size)
      : pos(static_cast<const char*>(buf))
      , nrem(size)
  {}

  void read_bytes(void* out, size_t n)
  {
    if (n > nrem)
    {
      throw std::runtime_error("aot blob truncated");
    }
    std::memcpy(out, pos, n);
    pos += n;
    nrem -= n;
  }

  template <class T>
  T read_pod()
  {
    static_assert(std::is_trivially_copyable_v<T>, "read_pod requires trivially-copyable type");
    T v;
    read_bytes(&v, sizeof(T));
    return v;
  }

  // Reads a length-prefixed string and returns a fresh new[]'d copy
  // (always nul-terminated). Length=0 returns nullptr.
  char* read_cstring_dup()
  {
    const uint64_t n = read_pod<uint64_t>();
    if (n == 0)
    {
      return nullptr;
    }
    if (n > nrem)
    {
      throw std::runtime_error("aot blob truncated (cstring)");
    }
    auto out = std::make_unique<char[]>(n + 1);
    std::memcpy(out.get(), pos, n);
    out[n] = '\0';
    pos += n;
    nrem -= n;
    return out.release();
  }

  // Reads a length-prefixed byte blob into a fresh new[]'d buffer.
  // Length=0 returns nullptr with *out_size=0.
  void read_blob_new(void** out_buf, size_t* out_size)
  {
    const uint64_t n = read_pod<uint64_t>();
    if (n > nrem)
    {
      throw std::runtime_error("aot blob truncated (blob)");
    }
    if (n == 0)
    {
      *out_buf  = nullptr;
      *out_size = 0;
      return;
    }
    auto out = std::make_unique<char[]>(n);
    std::memcpy(out.get(), pos, n);
    pos += n;
    nrem -= n;
    *out_buf  = out.release();
    *out_size = n;
  }

  // Reads a length-prefixed POD blob directly into an existing pointer
  // (allocated by the caller as new T). Used for runtime_policy where the
  // target type's allocator must match the algorithm's cleanup path.
  void read_into(void* dest, size_t expected_size)
  {
    const uint64_t n = read_pod<uint64_t>();
    if (n != expected_size)
    {
      throw std::runtime_error("aot blob runtime_policy size mismatch");
    }
    if (n > 0)
    {
      read_bytes(dest, n);
    }
  }

  size_t remaining() const noexcept
  {
    return nrem;
  }
};

// Writes the standard blob header.
inline void write_header(buffer_writer& w, cccl_aot_algo_t algo_tag, cccl_payload_kind_t kind, int cc)
{
  blob_header h{};
  std::memcpy(h.magic, k_blob_magic, sizeof(k_blob_magic));
  h.algo_tag       = static_cast<uint32_t>(algo_tag);
  h.format_version = k_format_version;
  h.cccl_version   = static_cast<uint64_t>(CCCL_C_PARALLEL_VERSION);
  h.payload_kind   = static_cast<uint32_t>(kind);
  h.cc             = static_cast<uint32_t>(cc);
  w.write_pod(h);
}

// Reads + validates a blob header. Throws on magic / algo_tag / format_version /
// cccl_version mismatch. Returns the parsed header for the caller to use
// (payload_kind, cc).
inline blob_header read_and_validate_header(buffer_reader& r, cccl_aot_algo_t expected_algo)
{
  const auto h = r.read_pod<blob_header>();
  if (std::memcmp(h.magic, k_blob_magic, sizeof(k_blob_magic)) != 0)
  {
    throw std::runtime_error("aot blob: bad magic");
  }
  if (h.algo_tag != static_cast<uint32_t>(expected_algo))
  {
    throw std::runtime_error("aot blob: wrong algorithm");
  }
  if (h.format_version != k_format_version)
  {
    throw std::runtime_error("aot blob: unsupported format version");
  }
  if (h.cccl_version != static_cast<uint64_t>(CCCL_C_PARALLEL_VERSION))
  {
    throw std::runtime_error(std::format(
      "aot blob: CCCL C parallel version mismatch (blob={}, current={})", h.cccl_version, CCCL_C_PARALLEL_VERSION));
  }
  if (h.payload_kind != CCCL_PAYLOAD_LTOIR && h.payload_kind != CCCL_PAYLOAD_CUBIN)
  {
    throw std::runtime_error("aot blob: unknown payload kind");
  }
  return h;
}

// Serializes a cccl_type_info as a fixed POD record.
inline void write_type_info(buffer_writer& w, const cccl_type_info& t)
{
  w.write_pod<uint64_t>(static_cast<uint64_t>(t.size));
  w.write_pod<uint64_t>(static_cast<uint64_t>(t.alignment));
  w.write_pod<uint32_t>(static_cast<uint32_t>(t.type));
}

inline cccl_type_info read_type_info(buffer_reader& r)
{
  cccl_type_info t{};
  t.size            = static_cast<size_t>(r.read_pod<uint64_t>());
  t.alignment       = static_cast<size_t>(r.read_pod<uint64_t>());
  const auto type_v = r.read_pod<uint32_t>();
  if (type_v > static_cast<uint32_t>(CCCL_BOOLEAN))
  {
    throw std::runtime_error(std::format("aot blob: invalid type enum ({})", type_v));
  }
  t.type = static_cast<cccl_type_enum>(type_v);
  return t;
}
} // namespace cccl::aot
