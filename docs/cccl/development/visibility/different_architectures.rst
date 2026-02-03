.. _cccl-development-visibility-different-architectures:

Linking TUs compiled with different architectures
--------------------------------------------------

Consider the following simple library:

.. code-block:: cpp

    template <int... Archs>
    __host__ __device__ constexpr int sum_archs() noexcept {
      return (Archs + ... + 0);
    }

    // kernel with architecture dependent symbol name and functionality
    template <class T, auto Archs = sum_archs<__CUDA_ARCH_LIST__>()>
    __global__ void kernel(T *val) {
      *val = sum_archs<__CUDA_ARCH_LIST__>();
    }

    __attribute__((visibility("hidden"))) inline int use_kernel() {
      int *d_val{};
      cudaMalloc(&d_val, sizeof(d_val));
      kernel<<<1, 1>>>(d_val);
      int ret;
      if (cudaMemcpy(&ret, d_val, sizeof(size_t), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        std::printf("c: FAILED to copy from device to host\n");
      }
      return ret;
    }

    template <class T = int>
    struct some_class_with_kernel {
      T val_;

      some_class_with_kernel();
      __forceinline__ some_class_with_kernel(T) { val_ = use_kernel(); }
    };

We have a kernel that does some architecture dependent work. This could be relying on some hardware feature that is
dependent on the current architecture.

.. code-block:: cpp

  #include "kernel.cuh"

  int main() {
    some_class_with_kernel with_inline{1};
    std::printf("a: value of class with inlined constructor: %d\n",
                with_inline.val_);

    some_class_with_kernel from_library{};
    std::printf("a: value of class with constructor from library: %d\n",
                from_library.val_);
  }

Importantly, one of the constructors for that class is put into a shared library, whereas the other one happens to be
inlined. If a user now links two different libraries, the outcome of the initialization of ``some_class_with_kernel``
will depend on whether the inlined constructor is called and which of the libraries is loaded first by the linker.

Even worse, the state of a class depends on whether the constructor has been inlined or not and the order in which
the linker loads the libraries.

.. code-block:: cmake

  project(CUBVisDifferentArchitectures CUDA CXX)

  add_library(cubvis_different_architectures_lib_a SHARED tu_a.cu)
  set_target_properties(cubvis_different_architectures_lib_a PROPERTIES CUDA_ARCHITECTURES "86;90a")

  add_library(cubvis_different_architectures_lib_b SHARED tu_b.cu)
  set_target_properties(cubvis_different_architectures_lib_b PROPERTIES CUDA_ARCHITECTURES "75;86;90a")

  add_executable(cubvis_different_architectures main.cu)
  set_target_properties(cubvis_different_architectures PROPERTIES CUDA_ARCHITECTURES "75;86")

  target_link_libraries(cubvis_different_architectures PRIVATE
    cubvis_different_architectures_lib_a
    cubvis_different_architectures_lib_b)

  add_executable(cubvis_different_architectures_switched main.cu)
  set_target_properties(cubvis_different_architectures_switched PROPERTIES CUDA_ARCHITECTURES "75;86")

  target_link_libraries(cubvis_different_architectures_switched PRIVATE
    cubvis_different_architectures_lib_b
    cubvis_different_architectures_lib_a)

Execution the two libraries will result in the following:

.. code-block::

  ./different_architectures/different_architectures
  a: value of class with inlined constructor: 1610       <<<--- from main
  a: value of class with constructor from library: 1760  <<<--- from lib_a

  ./different_architectures/different_architectures_switched
  a: value of class with inlined constructor: 1610       <<<--- from main
  a: value of class with constructor from library: 2510  <<<--- from lib_b


One solution would be to bake the architectures into the symbol name of the class, either via a defaulted template
argument or an inline namespace. That way the usage of the non-inlined kernel would result in a linker error, because
we did not provide a matching implementation.

.. code-block::

  tmpxft_00048dff_00000000-6_main.compute_86.cudafe1.cpp:(.text.startup+0xc0):
  undefined reference to `some_class_with_kernel<int, 5120ul>::some_class_with_kernel()'

However, if all the functionality is within a non-inlined function we would still get different results, because all
kernel definitions would be internal to the respective library.

.. code-block:: cpp

  // In tu_a.cu and tu_b.cu
  void non_inlined_function() {
    some_class_with_kernel with_inline{1};
    std::printf("a: value of class with inlined constructor: %d\n",
                with_inline.val_);

    some_class_with_kernel from_library{};
    std::printf("a: value of class with constructor from library: %d\n",
                from_library.val_);
  }

  // In main.cu
  #include "kernel.cuh"

  void non_inlined_function();

  int main() {
    some_class_with_kernel with_inline{1};
    std::printf("a: value of class with inlined constructor: %d\n",
                with_inline.val_);

    non_inlined_function();
  }

Executing this binary will give us again:

.. code-block::

  ./different_architectures/different_architectures
  a: value of class with inlined constructor: 1610       <<<--- from main
  a: value of class with inlined constructor: 1760       <<<--- from lib_a
  a: value of class with constructor from library: 1760  <<<--- from lib_a

  ./different_architectures/different_architectures_switched
  a: value of class with inlined constructor: 1610       <<<--- from main
  a: value of class with inlined constructor: 2510       <<<--- from lib_a
  a: value of class with constructor from library: 2510  <<<--- from lib_b

So there is not functional way we can solve this problem generically, because the moment a user actually uses any type
of function that executes a kernel and puts that function into a shared library there is no guarantee which function
is selected. The same happens if the user builds a type

.. code-block:: cpp

  class user_defined_with_kernel {
    some_class_with_kernel val;

    user_defined_with_kernel();
    __forceinline__ user_defined_with_kernel(T input) : val(input)
    {}
  };

  void function_that_uses_kernel_inside();

If ``user_defined_with_kernel`` is ever baked into a library we would be back with the same exact problem,
just one layer up. The user would need to know that ``some_class_with_kernel`` uses a kernel and then annotate *their*
classes and functions appropriately. This is neither realistic nor feasible.

Lets circle back to the previous statement: ``This is bad.`` Is it really though?

Lets look at the prime example ``thrust::device_vector``, which uses a kernel for initialization. What happens if we
accidentally run the kernel from another shared library compiled with different architectures? Worst case we are
eating some performance regressions because the kernel will not utilize advanced features of a new architecture,
but in the end the result of calling that kernel will not change the outcome.

This is because the kernel call is consistent *within* each library. As long as the user facing API does not rely on
specific internals of a kernel to be called -which it should not-, then any of the two libraries will do.

Finally, the architectures that are passed around in ``__CUDA_ARCH_LIST__`` do *not* discriminate architecture families.
There is currently no programmatic way to discriminate a library that has been compiled for ``SM90a`` from one that was
compiled for ``SM90``. This is because the architecture specific macros are only available on device not on host.
