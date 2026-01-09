//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Pretty printing utilities
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/utility/scope_guard.cuh>

#include <cmath>
#include <iomanip>
#include <sstream>

namespace cuda::experimental::stf
{
/**
 * @brief Convert a size into a human readable string
 */
inline ::std::string pretty_print_bytes(size_t bytes)
{
  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  size_t size         = sizeof(units) / sizeof(char*);
  int i               = 0;

  double pretty_size = static_cast<double>(bytes);
  while (pretty_size >= 1024.0 && static_cast<size_t>(i) < size - 1)
  {
    pretty_size /= 1024.0;
    ++i;
  }

  ::std::ostringstream out;
  out << ::std::fixed << ::std::setprecision(2) << pretty_size << ' ' << units[i];
  return out.str();
}

namespace reserved
{
/**
 * A trait class to have the specifier to display the T type.
 */
template <typename T>
struct FormatSpecifier
{
  static constexpr const char* value = "%f "; // Default for floating point
  static constexpr const char* name  = "float"; // Default for floating point
};

template <>
struct FormatSpecifier<int>
{
  static constexpr const char* value = "%d "; // For int
  static constexpr const char* name  = "int";
};

template <>
struct FormatSpecifier<unsigned int>
{
  static constexpr const char* value = "%u "; // For unsigned int
  static constexpr const char* name  = "unsigned int";
};

template <>
struct FormatSpecifier<uint64_t>
{
  static constexpr const char* value = "%llu "; // For uint64_t, assuming LLP64 or LP64 model
  static constexpr const char* name  = "uint64_t";
};
} // end namespace reserved

/**
 * @brief Writes an `mdspan` to a VTK file.
 *
 * This function writes an `mdspan` to a VTK file, which is useful for visualization and debugging.
 *
 * @tparam P The template parameter pack for mdspan
 * @param s The input mdspan to be written
 * @param filename The output file name where the VTK data will be written
 */
template <typename mdspan_like>
void mdspan_to_vtk(mdspan_like s, const ::std::string& filename)
{
  fprintf(stderr, "Writing slice of size to file %s\n", filename.c_str());
  FILE* f = EXPECT(fopen(filename.c_str(), "w+") != nullptr);
  SCOPE(exit)
  {
    EXPECT(fclose(f) != -1);
  };

  EXPECT(fprintf(f, "# vtk DataFile Version 2.0\noutput\nASCII\n") != -1);

  size_t nx = 1;
  size_t ny = 1;
  size_t nz = 1;

  if constexpr (s.rank() > 0)
  {
    nx = s.extent(0);
  }

  if constexpr (s.rank() > 1)
  {
    ny = s.extent(1);
  }

  if constexpr (s.rank() > 2)
  {
    nz = s.extent(2);
  }

  EXPECT(fprintf(f, "DATASET STRUCTURED_POINTS\n") != -1);
  EXPECT(fprintf(f, "DIMENSIONS %zu %zu %zu\n", nx, ny, nz) != -1);
  EXPECT(fprintf(f, "ORIGIN 0 0 0\n") != -1);
  EXPECT(fprintf(f, "SPACING 1 1 1\n") != -1);
  EXPECT(fprintf(f, "POINT_DATA %zu\n", s.size()) != -1);
  EXPECT(fprintf(f, "SCALARS value float\n") != -1);
  EXPECT(fprintf(f, "LOOKUP_TABLE default\n") != -1);

  static_assert(s.rank() <= 3 && s.rank() > 0);

  if constexpr (s.rank() == 1)
  {
    for (size_t x = 0; x < nx; x++)
    {
      EXPECT(fprintf(f, "%f ", s(x)) != -1);
    }
  }

  if constexpr (s.rank() == 2)
  {
    for (size_t y = 0; y < ny; y++)
    {
      for (size_t x = 0; x < nx; x++)
      {
        EXPECT(fprintf(f, "%f ", s(x, y)) != -1);
      }
      EXPECT(fprintf(f, "\n") != -1);
    }
  }

  if constexpr (s.rank() == 3)
  {
    for (size_t z = 0; z < nz; z++)
    {
      for (size_t y = 0; y < ny; y++)
      {
        for (size_t x = 0; x < nx; x++)
        {
          EXPECT(fprintf(f, "%f ", s(x, y, z)) != -1);
        }
        EXPECT(fprintf(f, "\n") != -1);
      }
    }
  }

  //    for (size_t y = 0; y < 4; y++)
  //    {
  //        for (size_t x = 0; x < 4; x++)
  //        {
  //            fprintf(stderr, "%f ", s(y, x));
  //        }
  //        fprintf(stderr, "\n");
  //    }
}

/**
 * @brief Print a formatted slice to a file or standard error output.
 *
 * @tparam T The type of data contained in the slice
 * @tparam dimensions The rank of the slice
 * @param s The slice object to be printed
 * @param text The descriptive text to be printed before the slice data
 * @param f The file pointer where the slice data should be printed (default: standard error output)
 *
 * The output is prefixed by `text` followed by a newline.
 */
template <typename mdspan_like>
void mdspan_print(mdspan_like s, const ::std::string& text, FILE* f = stderr)
{
  using T = typename mdspan_like::value_type;

  fprintf(f, "%s\n", text.c_str());
  const char* format = reserved::FormatSpecifier<T>::value;

  size_t nx = 1;
  size_t ny = 1;
  size_t nz = 1;

  static constexpr size_t dimensions = s.rank();

  if constexpr (dimensions > 0)
  {
    nx = s.extent(0);
  }

  if constexpr (dimensions > 1)
  {
    ny = s.extent(1);
  }

  if constexpr (dimensions > 2)
  {
    nz = s.extent(2);
  }

  static_assert(dimensions <= 3 && dimensions > 0);

  if constexpr (dimensions == 1)
  {
    for (size_t x = 0; x < nx; x++)
    {
      EXPECT(fprintf(f, format, s(x)) != -1);
    }
  }

  if constexpr (dimensions == 2)
  {
    for (size_t y = 0; y < ny; y++)
    {
      for (size_t x = 0; x < nx; x++)
      {
        EXPECT(fprintf(f, format, s(x, y)) != -1);
      }
      EXPECT(fprintf(f, "\n") != -1);
    }
  }

  if constexpr (dimensions == 3)
  {
    for (size_t z = 0; z < nz; z++)
    {
      for (size_t y = 0; y < ny; y++)
      {
        for (size_t x = 0; x < nx; x++)
        {
          EXPECT(fprintf(f, format, s(x, y, z)) != -1);
        }
        EXPECT(fprintf(f, "\n") != -1);
      }
    }
  }
}
} // end namespace cuda::experimental::stf
