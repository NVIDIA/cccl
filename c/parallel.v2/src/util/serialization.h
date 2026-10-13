//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cccl/c/serialization.h>
#include <cccl/c/types.h>

namespace cccl::serialization_v2
{
// Opaque 8-byte marker identifying a CCCL C parallel v2 (HostJIT)
// serialization blob. Deliberately distinct from v1's "CCCLSER1" so a v1
// blob can never be mistaken for a v2 one (or vice versa) — the two
// backends' payloads are fundamentally different formats (a driver-loadable
// device blob for v1, a full native shared library for v2).
inline constexpr char k_blob_magic[8] = {'C', 'C', 'C', 'L', 'S', 'E', 'V', '2'};

// Which OS/CPU architecture a blob's shared-library payload was built for.
// Rejected before ever calling dlopen/LoadLibrary on the payload — see
// read_blob(). Cross-OS/cross-arch portability is explicitly out of scope
// (inherent to native shared libraries); same-OS-family portability
// (Linux->Linux, Windows->Windows, regardless of exact CUDA Toolkit
// install path) is the supported case — see hostjit::JITCompiler::loadFromBytes.
enum class os_arch_tag_t : uint32_t
{
  LINUX_X86_64  = 0,
  LINUX_AARCH64 = 1,
  WIN_X86_64    = 2,
};

// The os_arch_tag_t for the machine this code is compiled for.
constexpr os_arch_tag_t current_os_arch_tag()
{
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
  return os_arch_tag_t::WIN_X86_64;
#elif defined(__linux__) && (defined(__aarch64__) || defined(_M_ARM64))
  return os_arch_tag_t::LINUX_AARCH64;
#elif defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))
  return os_arch_tag_t::LINUX_X86_64;
#else
#  error "cccl.c.parallel.v2 serialization: unsupported OS/architecture"
#endif
}

// Fixed-layout header at the start of every v2 blob. Packed POD; layout is
// part of the on-disk format, do not reorder. No package-version field is
// carried here: that compatibility check is handled by the Python
// _serialization/codec.py layer (the same package-version gate used for
// v1), not this C layer — matching v1's util/serialization.h design.
#pragma pack(push, 1)
struct blob_header
{
  char magic[8]; // k_blob_magic
  uint32_t algo_tag; // cccl_serialization_algo_v2_t
  uint32_t os_arch_tag; // os_arch_tag_t
  uint32_t cc; // cc_major*10 + cc_minor
  uint32_t num_symbols; // number of exported symbol names that follow the header
};
#pragma pack(pop)
static_assert(sizeof(blob_header) == 24, "blob_header layout must be stable");

// Serializes: header, then num_symbols length-prefixed (u32 length + bytes,
// no NUL terminator) symbol-name strings in the order _load() must dlsym
// them, then an algorithm-specific `extra` metadata span (u32 length +
// bytes — e.g. reduce packs {accumulator_size, determinism} here; each
// algorithm's _serialize/_deserialize owns the encoding, this layer just
// carries the bytes), then payload_size (u64) + payload bytes (the
// compiled .so/.dll).
inline std::vector<char> write_blob(
  cccl_serialization_algo_v2_t algo,
  int cc,
  const std::vector<std::string>& symbol_names,
  const std::vector<char>& extra,
  const void* payload,
  size_t payload_size)
{
  std::vector<char> out;
  out.reserve(sizeof(blob_header) + payload_size + extra.size() + 64);

  blob_header header{};
  std::memcpy(header.magic, k_blob_magic, sizeof(k_blob_magic));
  header.algo_tag    = static_cast<uint32_t>(algo);
  header.os_arch_tag = static_cast<uint32_t>(current_os_arch_tag());
  header.cc          = static_cast<uint32_t>(cc);
  header.num_symbols = static_cast<uint32_t>(symbol_names.size());

  auto append = [&](const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    out.insert(out.end(), c, c + n);
  };

  append(&header, sizeof(header));
  for (const auto& name : symbol_names)
  {
    uint32_t len = static_cast<uint32_t>(name.size());
    append(&len, sizeof(len));
    append(name.data(), name.size());
  }
  uint32_t extra_len = static_cast<uint32_t>(extra.size());
  append(&extra_len, sizeof(extra_len));
  append(extra.data(), extra.size());
  uint64_t psize = static_cast<uint64_t>(payload_size);
  append(&psize, sizeof(psize));
  append(payload, payload_size);
  return out;
}

// Result of a successful read_blob(): parsed metadata plus a view INTO the
// input buffer (payload/payload's lifetime is tied to the caller's buffer —
// copy out before that buffer is freed/goes out of scope).
struct parsed_blob
{
  cccl_serialization_algo_v2_t algo_tag;
  int cc;
  std::vector<std::string> symbol_names;
  std::vector<char> extra; // algorithm-specific metadata, see write_blob()
  const char* payload;
  size_t payload_size;
};

// Parses and validates a blob written by write_blob(). Throws
// std::runtime_error with a descriptive message on any validation failure
// (bad magic, wrong algorithm, wrong OS/architecture, truncated buffer) —
// callers must validate before ever calling dlopen/LoadLibrary on the
// payload. Deliberately does NOT check `cc` against the current device:
// that would make deserialize() require a live GPU, which breaks the
// no-GPU / cross-cc AoT use case — a cc mismatch instead surfaces later, at
// kernel-launch time, as cudaErrorNoKernelImageForDevice. An explicit early
// cc check belongs at _load() time instead (where a live device is a
// reasonable thing to assume), not here.
inline parsed_blob read_blob(cccl_serialization_algo_v2_t expected_algo, const void* buf, size_t size)
{
  const char* p = static_cast<const char*>(buf);
  size_t pos    = 0;
  auto need     = [&](size_t n) {
    if (n > size - pos)
    {
      throw std::runtime_error("cccl.c.parallel.v2 serialization: truncated blob");
    }
  };

  need(sizeof(blob_header));
  blob_header header;
  std::memcpy(&header, p + pos, sizeof(header));
  pos += sizeof(header);

  if (std::memcmp(header.magic, k_blob_magic, sizeof(k_blob_magic)) != 0)
  {
    throw std::runtime_error("cccl.c.parallel.v2 serialization: bad magic (not a v2 CCCL C parallel blob)");
  }
  if (header.algo_tag != static_cast<uint32_t>(expected_algo))
  {
    throw std::runtime_error("cccl.c.parallel.v2 serialization: algorithm tag mismatch");
  }
  if (header.os_arch_tag != static_cast<uint32_t>(current_os_arch_tag()))
  {
    throw std::runtime_error(
      "cccl.c.parallel.v2 serialization: blob was built for a different OS/architecture than this machine "
      "(cccl.c.parallel.v2 AoT blobs are portable within the same OS family and CPU architecture, e.g. "
      "Linux x86_64 -> Linux x86_64, but not across OS or architecture)");
  }

  parsed_blob result{};
  result.algo_tag = static_cast<cccl_serialization_algo_v2_t>(header.algo_tag);
  result.cc       = static_cast<int>(header.cc);

  for (uint32_t i = 0; i < header.num_symbols; ++i)
  {
    need(sizeof(uint32_t));
    uint32_t len;
    std::memcpy(&len, p + pos, sizeof(len));
    pos += sizeof(len);
    need(len);
    result.symbol_names.emplace_back(p + pos, len);
    pos += len;
  }

  need(sizeof(uint32_t));
  uint32_t extra_len;
  std::memcpy(&extra_len, p + pos, sizeof(extra_len));
  pos += sizeof(extra_len);
  need(extra_len);
  result.extra.assign(p + pos, p + pos + extra_len);
  pos += extra_len;

  need(sizeof(uint64_t));
  uint64_t psize;
  std::memcpy(&psize, p + pos, sizeof(psize));
  pos += sizeof(psize);
  need(static_cast<size_t>(psize));
  result.payload      = p + pos;
  result.payload_size = static_cast<size_t>(psize);

  return result;
}

// Packs an algorithm's `extra` metadata span for write_blob(). Fields are
// written in call order with no padding, so the byte layout is exactly the
// sequence of put() calls. Pair with extra_reader, which must read the same
// sequence back.
class extra_writer
{
public:
  // Fixed-width arithmetic types only (cast enums explicitly at the call site
  // so the on-disk width never depends on the enum's underlying type).
  template <typename T>
  extra_writer& put(T value)
  {
    static_assert(std::is_arithmetic_v<T>, "extra_writer::put takes fixed-width arithmetic types");
    const char* p = reinterpret_cast<const char*>(&value);
    bytes_.insert(bytes_.end(), p, p + sizeof(T));
    return *this;
  }

  // {size (u64), alignment (u64), type (u32)}
  extra_writer& put_type(const cccl_type_info& t)
  {
    return put<uint64_t>(t.size).put<uint64_t>(t.alignment).put<uint32_t>(static_cast<uint32_t>(t.type));
  }

  std::vector<char> take()
  {
    return std::move(bytes_);
  }

private:
  std::vector<char> bytes_;
};

// Reads back what extra_writer wrote. Never throws: reading past the end
// yields zeros and latches a failure, so a caller can read every field and
// then check ok() once (which also requires the span to be fully consumed).
class extra_reader
{
public:
  explicit extra_reader(const std::vector<char>& bytes)
      : bytes_(bytes)
  {}
  // The reader only views the bytes; refuse temporaries that would dangle.
  extra_reader(std::vector<char>&&) = delete;

  template <typename T>
  T get()
  {
    static_assert(std::is_arithmetic_v<T>, "extra_reader::get takes fixed-width arithmetic types");
    T value{};
    if (sizeof(T) > bytes_.size() - pos_)
    {
      failed_ = true;
      return value;
    }
    std::memcpy(&value, bytes_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return value;
  }

  cccl_type_info get_type()
  {
    const auto size      = get<uint64_t>();
    const auto alignment = get<uint64_t>();
    const auto type      = get<uint32_t>();
    return cccl_type_info{static_cast<size_t>(size), static_cast<size_t>(alignment), static_cast<cccl_type_enum>(type)};
  }

  // True iff every read succeeded and no bytes are left over.
  bool ok() const
  {
    return !failed_ && pos_ == bytes_.size();
  }

private:
  const std::vector<char>& bytes_;
  size_t pos_  = 0;
  bool failed_ = false;
};

// Hands a finished blob to the caller as a `new char[]` (the buffer the
// *_serialize C API documents as caller-owned).
inline void release_blob(const std::vector<char>& blob, void** out_buf, size_t* out_size)
{
  auto* buf = new char[blob.size()];
  std::memcpy(buf, blob.data(), blob.size());
  *out_buf  = buf;
  *out_size = blob.size();
}
} // namespace cccl::serialization_v2
