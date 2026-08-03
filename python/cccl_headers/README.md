<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# cccl-headers

`cccl-headers` provides the C++ headers and relocatable CMake package files for
libcu++, CUB, Thrust, and CUDA Experimental (CUDAX). It does not install a CUDA
Toolkit or any Python compiler runtime.

```python
from cuda.cccl.headers import get_include_paths

include_paths = get_include_paths()
compiler_include_arguments = [
    f"-I{path}" for path in include_paths.as_tuple() if path is not None
]
```

`get_include_paths()` uses `cuda-pathfinder` to locate the CUDA Toolkit header
directory. The remaining paths point to the headers bundled in this wheel.
