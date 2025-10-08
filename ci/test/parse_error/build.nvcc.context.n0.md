📝 `heterogeneous_iterator.cuh.cu:5`: `error #549-D: variable "i" is used befor...`

📍 Location: `cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__container/heterogeneous_iterator.cuh.cu:5`

🎯 Target Name: cudax.cpp17.headers.basic.no_stf

🔍 Full Error:

<pre>
  FAILED: cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__container/heterogeneous_iterator.cuh.cu.o
  /usr/bin/sccache /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++ -DCCCL_ENABLE_ASSERTIONS -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -D_CCCL_NO_SYSTEM_HEADER -I/home/coder/cccl/cudax/include -I/home/coder/cccl/lib/cmake/libcudacxx/../../../libcudacxx/include -I/home/coder/cccl/lib/cmake/cub/../../../cub -I/home/coder/cccl/lib/cmake/thrust/../../../thrust -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_75,code=[sm_75]" "--generate-code=arch=compute_80,code=[sm_80]" "--generate-code=arch=compute_90,code=[sm_90]" "--generate-code=arch=compute_100,code=[sm_100]" "--generate-code=arch=compute_110,code=[sm_110]" "--generate-code=arch=compute_120,code=[sm_120]" "--generate-code=arch=compute_120,code=[compute_120]" -Xcudafe=--display_error_number -Wno-deprecated-gpu-targets -Xcudafe=--promote_warnings -Wreorder -Xcompiler=-Werror -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wreorder -Xcompiler=-Winit-self -Xcompiler=-Woverloaded-virtual -Xcompiler=-Wcast-qual -Xcompiler=-Wpointer-arith -Xcompiler=-Wvla -Xcompiler=-Wno-gnu-line-marker -Xcompiler=-Wno-gnu-zero-variadic-macro-arguments -Xcompiler=-Wno-unused-function -Xcompiler=-Wno-noexcept-type -MD -MT cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__container/heterogeneous_iterator.cuh.cu.o -MF cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__container/heterogeneous_iterator.cuh.cu.o.d -x cu -c /home/coder/cccl/build/cuda13.0-gcc11/cudax-cpp17/cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__container/heterogeneous_iterator.cuh.cu -o cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__container/heterogeneous_iterator.cuh.cu.o
  /home/coder/cccl/build/cuda13.0-gcc11/cudax-cpp17/cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__container/heterogeneous_iterator.cuh.cu(5): error #549-D: variable "i" is used before its value is set
      return i;
             ^

  Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

  1 error detected in the compilation of "/home/coder/cccl/build/cuda13.0-gcc11/cudax-cpp17/cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__container/heterogeneous_iterator.cuh.cu".
</pre>

📝 `type_traits.cuh.cu:5`: `error #549-D: variable "i" is used before its value...`

📍 Location: `cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__detail/type_traits.cuh.cu:5`

🎯 Target Name: cudax.cpp17.headers.basic.no_stf

🔍 Full Error:

<pre>
  FAILED: cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__detail/type_traits.cuh.cu.o
  /usr/bin/sccache /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++ -DCCCL_ENABLE_ASSERTIONS -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -D_CCCL_NO_SYSTEM_HEADER -I/home/coder/cccl/cudax/include -I/home/coder/cccl/lib/cmake/libcudacxx/../../../libcudacxx/include -I/home/coder/cccl/lib/cmake/cub/../../../cub -I/home/coder/cccl/lib/cmake/thrust/../../../thrust -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_75,code=[sm_75]" "--generate-code=arch=compute_80,code=[sm_80]" "--generate-code=arch=compute_90,code=[sm_90]" "--generate-code=arch=compute_100,code=[sm_100]" "--generate-code=arch=compute_110,code=[sm_110]" "--generate-code=arch=compute_120,code=[sm_120]" "--generate-code=arch=compute_120,code=[compute_120]" -Xcudafe=--display_error_number -Wno-deprecated-gpu-targets -Xcudafe=--promote_warnings -Wreorder -Xcompiler=-Werror -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wreorder -Xcompiler=-Winit-self -Xcompiler=-Woverloaded-virtual -Xcompiler=-Wcast-qual -Xcompiler=-Wpointer-arith -Xcompiler=-Wvla -Xcompiler=-Wno-gnu-line-marker -Xcompiler=-Wno-gnu-zero-variadic-macro-arguments -Xcompiler=-Wno-unused-function -Xcompiler=-Wno-noexcept-type -MD -MT cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__detail/type_traits.cuh.cu.o -MF cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__detail/type_traits.cuh.cu.o.d -x cu -c /home/coder/cccl/build/cuda13.0-gcc11/cudax-cpp17/cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__detail/type_traits.cuh.cu -o cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__detail/type_traits.cuh.cu.o
  /home/coder/cccl/build/cuda13.0-gcc11/cudax-cpp17/cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__detail/type_traits.cuh.cu(5): error #549-D: variable "i" is used before its value is set
      return i;
             ^

  Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

  1 error detected in the compilation of "/home/coder/cccl/build/cuda13.0-gcc11/cudax-cpp17/cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__detail/type_traits.cuh.cu".
</pre>

📝 `common.cuh.cu:5`: `error #549-D: variable "i" is used before its value is set`

📍 Location: `cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__algorithm/common.cuh.cu:5`

🎯 Target Name: cudax.cpp17.headers.basic.no_stf

🔍 Full Error:

<pre>
  FAILED: cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__algorithm/common.cuh.cu.o
  /usr/bin/sccache /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/usr/bin/g++ -DCCCL_ENABLE_ASSERTIONS -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -D_CCCL_NO_SYSTEM_HEADER -I/home/coder/cccl/cudax/include -I/home/coder/cccl/lib/cmake/libcudacxx/../../../libcudacxx/include -I/home/coder/cccl/lib/cmake/cub/../../../cub -I/home/coder/cccl/lib/cmake/thrust/../../../thrust -O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_75,code=[sm_75]" "--generate-code=arch=compute_80,code=[sm_80]" "--generate-code=arch=compute_90,code=[sm_90]" "--generate-code=arch=compute_100,code=[sm_100]" "--generate-code=arch=compute_110,code=[sm_110]" "--generate-code=arch=compute_120,code=[sm_120]" "--generate-code=arch=compute_120,code=[compute_120]" -Xcudafe=--display_error_number -Wno-deprecated-gpu-targets -Xcudafe=--promote_warnings -Wreorder -Xcompiler=-Werror -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wreorder -Xcompiler=-Winit-self -Xcompiler=-Woverloaded-virtual -Xcompiler=-Wcast-qual -Xcompiler=-Wpointer-arith -Xcompiler=-Wvla -Xcompiler=-Wno-gnu-line-marker -Xcompiler=-Wno-gnu-zero-variadic-macro-arguments -Xcompiler=-Wno-unused-function -Xcompiler=-Wno-noexcept-type -MD -MT cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__algorithm/common.cuh.cu.o -MF cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__algorithm/common.cuh.cu.o.d -x cu -c /home/coder/cccl/build/cuda13.0-gcc11/cudax-cpp17/cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__algorithm/common.cuh.cu -o cudax/CMakeFiles/cudax.cpp17.headers.basic.no_stf.dir/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__algorithm/common.cuh.cu.o
  /home/coder/cccl/build/cuda13.0-gcc11/cudax-cpp17/cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__algorithm/common.cuh.cu(5): error #549-D: variable "i" is used before its value is set
      return i;
             ^

  Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

  1 error detected in the compilation of "/home/coder/cccl/build/cuda13.0-gcc11/cudax-cpp17/cudax/headers/cudax.cpp17.headers.basic.no_stf/cuda/experimental/__algorithm/common.cuh.cu".
</pre>
