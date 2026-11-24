// SPDX-FileCopyrightText: Copyright (c) 2008-2009, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda/__cccl_config>

#if !_CCCL_CUDA_COMPILATION()

#  ifndef __host__
#    define __host__
#  endif

#  ifndef __device__
#    define __device__
#  endif

#endif
