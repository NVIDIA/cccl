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
inline constexpr char kBlobMagic[8] = {'C', 'C', 'C', 'L', 'A', 'O', 'T', '1'};

// Per-algorithm format version. Bump when the layout *for that algorithm*
// changes (e.g. a new field added to its build_result_t).
inline constexpr uint32_t kFormatVersion = 1;

// Fixed-layout header at the start of every blob. Packed POD; layout is
// part of the on-disk format, do not reorder.
struct blob_header
{
  char magic[8]; // kBlobMagic
  uint32_t algo_tag; // cccl_aot_algo_t
  uint32_t format_version; // kFormatVersion at write time
  uint64_t abi_hash; // identifies CCCL+CUB+struct ABI; mismatched → reject
  uint32_t payload_kind; // cccl_payload_kind_t
  uint32_t cc; // cc_major*10 + cc_minor
};
static_assert(sizeof(blob_header) == 32, "blob_header layout must be stable");

// Compile-time string hash; used to derive the ABI hash without pulling in
// runtime hashing machinery. Algorithm: FNV-1a 64-bit. Stable across builds.
constexpr uint64_t fnv1a64(std::string_view s) noexcept
{
  uint64_t h = 0xcbf29ce484222325ULL;
  for (char c : s)
  {
    h ^= static_cast<uint8_t>(c);
    h *= 0x100000001b3ULL;
  }
  return h;
}

// Mixes a numeric value into a running FNV-1a hash. Used to fold sizeof
// values (struct ABI footprint) into the blob's abi_hash.
constexpr uint64_t fnv1a64_mix(uint64_t h, uint64_t v) noexcept
{
  for (int i = 0; i < 8; ++i)
  {
    h ^= static_cast<uint8_t>(v & 0xff);
    h *= 0x100000001b3ULL;
    v >>= 8;
  }
  return h;
}

// Append-only byte buffer used by *_serialize implementations.
// Owns a std::vector<char> internally; release() hands back a heap buffer
// allocated with new[] (matching cccl_aot_buffer_free, which does delete[]).
class buffer_writer
{
  std::vector<char> data_;

public:
  void write_bytes(const void* p, size_t n)
  {
    if (n == 0)
    {
      return;
    }
    const char* src = static_cast<const char*>(p);
    data_.insert(data_.end(), src, src + n);
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
    return data_.size();
  }

  // Hands back ownership as a new[]'d buffer. After release() the writer is empty.
  void release(void** out_buf, size_t* out_size)
  {
    const size_t n = data_.size();
    auto p         = std::make_unique<char[]>(n);
    if (n > 0)
    {
      std::memcpy(p.get(), data_.data(), n);
    }
    data_.clear();
    *out_buf  = p.release();
    *out_size = n;
  }
};

// Bounds-checked byte buffer reader used by *_deserialize implementations.
// Borrows the input buffer; allocations it produces (via read_cstring_dup,
// read_blob_new) are owned by the caller.
class buffer_reader
{
  const char* p_;
  size_t remaining_;

public:
  buffer_reader(const void* buf, size_t size)
      : p_(static_cast<const char*>(buf))
      , remaining_(size)
  {}

  void read_bytes(void* out, size_t n)
  {
    if (n > remaining_)
    {
      throw std::runtime_error("aot blob truncated");
    }
    std::memcpy(out, p_, n);
    p_ += n;
    remaining_ -= n;
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
    if (n > remaining_)
    {
      throw std::runtime_error("aot blob truncated (cstring)");
    }
    auto out = std::make_unique<char[]>(n + 1);
    std::memcpy(out.get(), p_, n);
    out[n] = '\0';
    p_ += n;
    remaining_ -= n;
    return out.release();
  }

  // Reads a length-prefixed byte blob into a fresh new[]'d buffer.
  // Length=0 returns nullptr with *out_size=0.
  void read_blob_new(void** out_buf, size_t* out_size)
  {
    const uint64_t n = read_pod<uint64_t>();
    if (n > remaining_)
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
    std::memcpy(out.get(), p_, n);
    p_ += n;
    remaining_ -= n;
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
    return remaining_;
  }
};

// Writes the standard blob header. Caller supplies algo_tag + abi_hash.
inline void write_header(buffer_writer& w, cccl_aot_algo_t algo_tag, uint64_t abi_hash, cccl_payload_kind_t kind, int cc)
{
  blob_header h{};
  std::memcpy(h.magic, kBlobMagic, sizeof(kBlobMagic));
  h.algo_tag       = static_cast<uint32_t>(algo_tag);
  h.format_version = kFormatVersion;
  h.abi_hash       = abi_hash;
  h.payload_kind   = static_cast<uint32_t>(kind);
  h.cc             = static_cast<uint32_t>(cc);
  w.write_pod(h);
}

// Reads + validates a blob header. Throws on magic / algo_tag / version /
// abi_hash mismatch. Returns the parsed header for the caller to use
// (payload_kind, cc).
inline blob_header read_and_validate_header(buffer_reader& r, cccl_aot_algo_t expected_algo, uint64_t expected_abi_hash)
{
  const auto h = r.read_pod<blob_header>();
  if (std::memcmp(h.magic, kBlobMagic, sizeof(kBlobMagic)) != 0)
  {
    throw std::runtime_error("aot blob: bad magic");
  }
  if (h.algo_tag != static_cast<uint32_t>(expected_algo))
  {
    throw std::runtime_error("aot blob: wrong algorithm");
  }
  if (h.format_version != kFormatVersion)
  {
    throw std::runtime_error("aot blob: unsupported format version");
  }
  if (h.abi_hash != expected_abi_hash)
  {
    throw std::runtime_error("aot blob: ABI mismatch (CCCL/CUB version drift)");
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
  t.size      = static_cast<size_t>(r.read_pod<uint64_t>());
  t.alignment = static_cast<size_t>(r.read_pod<uint64_t>());
  t.type      = static_cast<cccl_type_enum>(r.read_pod<uint32_t>());
  return t;
}
} // namespace cccl::aot
