📝 `access_property_constexpr.pass.cpp:24`: `error: expected a ";"`

📍 Location: `libcudacxx/test/libcudacxx/cuda/annotated_ptr/access_property_constexpr.pass.cpp:24`

🎯 Target Name: cuda/annotated_ptr/access_property_constexpr.pass.cpp

🔍 Full Error:

<pre>
  ******************** TEST 'libcu++ :: cuda/annotated_ptr/access_property_constexpr.pass.cpp' FAILED ********************
  Command: ['/usr/bin/sccache', '/usr/local/cuda/bin/nvcc', '-o', '/home/coder/cccl/build/cuda12.9-llvm19/libcudacxx-cpp20/libcudacxx/test/libcudacxx/test/cuda/annotated_ptr/Output/access_property_constexpr.pass.cpp.o', '-x', 'cu', '/home/coder/cccl/libcudacxx/test/libcudacxx/cuda/annotated_ptr/access_property_constexpr.pass.cpp', '-c', '-std=c++20', '-ftemplate-depth=270', '-ccbin=/usr/bin/clang++', '-include', '/home/coder/cccl/libcudacxx/test/support/nasty_macros.h', '-I/home/coder/cccl/libcudacxx/include', '-D__STDC_FORMAT_MACROS', '-D__STDC_LIMIT_MACROS', '-D__STDC_CONSTANT_MACROS', '-Xcompiler', '-fno-rtti', '-I/home/coder/cccl/libcudacxx/test/support', '-D_CCCL_NO_SYSTEM_HEADER', '-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE', '-DCCCL_ENABLE_ASSERTIONS', '-DCCCL_ENABLE_OPTIONAL_REF', '-DCCCL_IGNORE_DEPRECATED_CPP_DIALECT', '-DLIBCUDACXX_IGNORE_DEPRECATED_ABI', '-include', '/home/coder/cccl/libcudacxx/test/libcudacxx/force_include.h', '--compiler-options=-Wall', '--compiler-options=-Wextra', '-Wno-deprecated-gpu-targets', '--extended-lambda', '-gencode=arch=compute_100,code=sm_100', '-gencode=arch=compute_120,code=sm_120', '-gencode=arch=compute_120,code=compute_120', '-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_80,code=sm_80', '-gencode=arch=compute_90,code=sm_90', '-Xcudafe', '--display_error_number', '-Werror=all-warnings', '-Xcompiler', '-Wno-user-defined-literals', '-Xcompiler', '-Wno-unused-parameter', '-Xcompiler', '-Wno-unused-local-typedefs', '-Xcompiler', '-Wno-deprecated-declarations', '-Xcompiler', '-Wno-noexcept-type', '-Xcompiler', '-Wno-unused-function', '-D_LIBCUDACXX_DISABLE_PRAGMA_GCC_SYSTEM_HEADER', '-c']
  Exit Code: 2
  Standard Error:
  --
  /home/coder/cccl/libcudacxx/test/libcudacxx/cuda/annotated_ptr/access_property_constexpr.pass.cpp(24): error: expected a ";"
      access_property b{a} + a - "foo";
                           ^

  1 error detected in the compilation of "/home/coder/cccl/libcudacxx/test/libcudacxx/cuda/annotated_ptr/access_property_constexpr.pass.cpp".
  --

  Compilation failed unexpectedly!
  ********************
</pre>
