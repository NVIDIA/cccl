//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cassert>

#include <cufile.h>

namespace cuda::experimental::cufile
{

//! @brief cuFile error codes (only success is defined by cuFile API)
enum class cu_file_error : int
{
  success = CU_FILE_SUCCESS
};

//! @brief cuFile operation codes
enum class cu_file_opcode : int
{
  read  = CUFILE_READ,
  write = CUFILE_WRITE
};

//! @brief cuFile status codes
enum class cu_file_status : int
{
  complete = CUFILE_COMPLETE,
  failed   = CUFILE_FAILED
};

//! @brief cuFile operation modes
enum class cu_file_mode : int
{
  batch = CUFILE_BATCH
};

//! @brief cuFile handle types
enum class cu_file_handle_type : int
{
  opaque_fd = CU_FILE_HANDLE_TYPE_OPAQUE_FD
};

//! @brief cuFile batch submit flags
enum class cu_file_batch_submit_flags : unsigned int
{
  none = 0u
};

inline constexpr cu_file_batch_submit_flags operator|(cu_file_batch_submit_flags lhs, cu_file_batch_submit_flags rhs)
{
  return static_cast<cu_file_batch_submit_flags>(static_cast<unsigned int>(lhs) | static_cast<unsigned int>(rhs));
}

inline constexpr cu_file_batch_submit_flags& operator|=(cu_file_batch_submit_flags& lhs, cu_file_batch_submit_flags rhs)
{
  lhs = lhs | rhs;
  return lhs;
}

inline constexpr cu_file_batch_submit_flags operator&(cu_file_batch_submit_flags lhs, cu_file_batch_submit_flags rhs)
{
  return static_cast<cu_file_batch_submit_flags>(static_cast<unsigned int>(lhs) & static_cast<unsigned int>(rhs));
}

inline constexpr bool any(cu_file_batch_submit_flags flags)
{
  return static_cast<unsigned int>(flags) != 0u;
}

// flags -> underlying C value
inline unsigned int to_c_enum(cu_file_batch_submit_flags flags)
{
  return static_cast<unsigned int>(flags);
}

//! @brief cuFile buffer registration flags (bitmask)
enum class cu_file_buf_register_flags : int
{
  none = 0
};

inline constexpr cu_file_buf_register_flags operator|(cu_file_buf_register_flags lhs, cu_file_buf_register_flags rhs)
{
  return static_cast<cu_file_buf_register_flags>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline constexpr cu_file_buf_register_flags& operator|=(cu_file_buf_register_flags& lhs, cu_file_buf_register_flags rhs)
{
  lhs = lhs | rhs;
  return lhs;
}

inline constexpr cu_file_buf_register_flags operator&(cu_file_buf_register_flags lhs, cu_file_buf_register_flags rhs)
{
  return static_cast<cu_file_buf_register_flags>(static_cast<int>(lhs) & static_cast<int>(rhs));
}

inline constexpr bool any(cu_file_buf_register_flags flags)
{
  return static_cast<int>(flags) != 0;
}

// flags -> underlying C value
inline int to_c_enum(cu_file_buf_register_flags flags)
{
  return static_cast<int>(flags);
}

// ===================== Converters (C++ <-> C) =====================

// cu_file_opcode -> CUfileOpcode_t
inline CUfileOpcode_t to_c_enum(cu_file_opcode op)
{
  switch (op)
  {
    case cu_file_opcode::read:
      return CUFILE_READ;
    case cu_file_opcode::write:
      return CUFILE_WRITE;
    default:
      assert(false && "Invalid cu_file_opcode");
      return CUFILE_READ;
  }
}

// CUfileStatus_t -> cu_file_status
inline cu_file_status to_cpp_enum(CUfileStatus_t status)
{
  switch (status)
  {
    case CUFILE_COMPLETE:
      return cu_file_status::complete;
    case CUFILE_FAILED:
      return cu_file_status::failed;
    default:
      assert(false && "Invalid cu_file_status");
      return cu_file_status::failed;
  }
}

// cu_file_status -> CUfileStatus_t
inline CUfileStatus_t to_c_enum(cu_file_status status)
{
  switch (status)
  {
    case cu_file_status::complete:
      return CUFILE_COMPLETE;
    case cu_file_status::failed:
      return CUFILE_FAILED;
    default:
      assert(false && "Invalid cu_file_status");
      return CUFILE_FAILED;
  }
}

// cu_file_mode -> CUfileBatchMode_t
inline CUfileBatchMode_t to_c_enum(cu_file_mode mode)
{
  switch (mode)
  {
    case cu_file_mode::batch:
      return CUFILE_BATCH;
    default:
      assert(false && "Invalid cu_file_mode");
      return CUFILE_BATCH;
  }
}

// cu_file_handle_type -> CUfileFileHandleType
inline CUfileFileHandleType to_c_enum(cu_file_handle_type type)
{
  switch (type)
  {
    case cu_file_handle_type::opaque_fd:
      return CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    default:
      assert(false && "Invalid cu_file_handle_type");
      return CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  }
}

// cu_file_error -> CUfileOpError
inline CUfileOpError to_c_enum(cu_file_error err)
{
  switch (err)
  {
    case cu_file_error::success:
      return CU_FILE_SUCCESS;
    default:
      assert(false && "Invalid cu_file_error");
      return CU_FILE_SUCCESS;
  }
}

// cuFile ssize_t result (negative error code) -> CUfileOpError
inline CUfileOpError to_c_enum_from_result(ssize_t result)
{
  assert(result < 0 && "Expected negative cuFile error code");
  // cuFile documents negative ssize_t values as CUfileOpError codes
  return static_cast<CUfileOpError>(result);
}

} // namespace cuda::experimental::cufile
