/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_NVHPC_SETTINGS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_NVHPC_SETTINGS_HPP_

extern "C" char* __nv_getenv(const char*);

namespace __nvhpc_std
{

struct __settings
{
  bool __runtime_fall_back;
};

inline __settings __stdblas_settings;

void __stdblas_get_setting(bool __default_value, char const* __env_variable, bool* __setting)
{
  *__setting = __default_value;

  char* __env = __nv_getenv(__env_variable);

  if (!__env)
  {
    return; // unset  NV_env_variable
  }

  if (*__env == '\0' // export NV_env_variable=
      || strcmp(__env, "1") == 0 // export NV_env_variable=1
  )
  {
    *__setting = true;
    return;
  }

  if (strcmp(__env, "0") == 0) // export NV_env_variable=0
  {
    *__setting = false;
  }
}

void __attribute__((constructor)) __stdblas_init()
{
  __stdblas_get_setting(false, "STDBLAS_RUNTIME_FALLBACK", &__stdblas_settings.__runtime_fall_back);
}

} // namespace __nvhpc_std

#endif
