``cuda.compute`` Developer Overview
===================================

This document provides an overview of the internal structure of
``cuda.compute``. At a high level, ``cuda.compute`` exposes CUDA C++
parallel algorithms through a Python API. Internally, it combines
Python-side operator compilation, CUDA C++ source generation, and
runtime just-in-time (JIT) compilation and linking.

We start with a simplified prototype. As we encounter the limitations
of that simplified model, we introduce the additional mechanisms
needed by the full implementation, referring to the relevant source
code where useful.

We begin with a minimal example that invokes a CUDA C++ kernel from
Python. In this simplified prototype, the kernel takes a single
integer argument and prints it.

.. code-block:: c++

    #include <cstdio>

    __global__ void kernel(int value) {
        std::printf("thread %d: %d\n", threadIdx.x, value);
    }

    extern "C" void launcher(int value) {
      kernel<<<1, 4>>>(value);
      cudaDeviceSynchronize();
    }

We can compile this code using ``nvcc``:

.. code-block:: bash

    nvcc -Xcompiler=-fPIC -x cu kernel.cu -shared -o libkernel.so

The resulting shared library exports the host function ``launcher``,
which we can call from Python using ``ctypes``:

.. code-block:: python

    import ctypes

    bindings = ctypes.CDLL('libkernel.so')
    bindings.launcher.argtypes = [ctypes.c_int]
    bindings.launcher(42)

Running that Python code produces:

.. code-block:: bash

    thread 0: 42
    thread 1: 42
    thread 2: 42
    thread 3: 42

The example above works because all of the behavior is fixed ahead of
time in the CUDA C++ source. The kernel and the operation it performs
are both known in advance.

A library primitive such as reduction is different. Its behavior
depends not only on the input type, but also on the operator being
applied. A practical Python API therefore cannot be limited to a
single built-in case such as summing ``float`` values. It needs to
support many data types and user-provided operators.

That means the CUDA C++ side must be able to invoke device code that
originates in Python. Reduction is a useful motivating example, but to
keep the mechanics simple we will start with a much smaller building
block: compiling a simple Python function and making it callable from
CUDA C++. The same technique later applies to user-provided reduction
operators.

We can compile such a Python function to PTX using
`Numba-CUDA <https://nvidia.github.io/numba-cuda/>`_ as follows:

.. code-block:: python

    import numba.cuda

    def op(value):
        return 2 * value

    ptx, _ = numba.cuda.compile(op, sig=numba.int32(numba.int32))

That'd give us the following PTX code:

.. code-block:: bash

    .visible .func  (.param .b32 func_retval0) op(.param .b32 op_param_0)
    {
            .reg .b32       %r<3>;


            ld.param.u32    %r1, [op_param_0];
           	shl.b32 	%r2, %r1, 1;
            st.param.b32    [func_retval0+0], %r2;
            ret;
    }

At this point, the Python function has been compiled to device code,
but the CUDA C++ side still needs a way to refer to it.

Conceptually, we would like to treat the operator as an externally
defined device function and call it from CUDA C++:

.. code-block:: c++

    #include <cstdio>

    extern "C" __device__ int op(int a); // defined in Python

    extern "C" __global__ void kernel(int value) {
        std::printf("thread %d: %d\n", threadIdx.x, op(value));
    }

    extern "C" void launcher(int value) {
      kernel<<<1, 4>>>(value);
      cudaDeviceSynchronize();
    }

This raises the next question: how do we combine device code produced
from Python with CUDA C++ code that calls it?

The difficulty is not just that the operator's implementation comes
from Python. The CUDA C++ side must also declare and call that
operator with the correct signature.

In the code above, the operator has the fixed signature ``int
op(int)``. A real API cannot assume that. The user might supply an
operator on ``float``, ``complex``, or some user-defined type, and the
generated CUDA C++ code has to match that interface exactly. In other
words, the declaration of ``op`` and the CUDA C++ source that calls it
depend on the user's types and operator signature.

That means the CUDA C++ side must be generated and compiled at
runtime. Using ``nvcc`` for that would make the API depend on an
external compiler toolchain being available on every user machine.
Instead, we use NVRTC, which is designed for runtime compilation of
CUDA C++.

Our Python code is now:

.. code-block:: python

    import ctypes
    import numba.cuda

    def op(value):
        return 2 * value

    ptx, _ = numba.cuda.compile(op, sig=numba.int32(numba.int32))

    bindings = ctypes.CDLL('./build/libkernel.so')
    bindings.launcher.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    bindings.launcher(42, ptx.encode('utf-8'), len(ptx))

Correspondingly, the C++ launcher now accepts the operator PTX as an
additional argument. Inside the launcher, the CUDA C++ kernel is now
assembled as a source string and compiled with NVRTC:

.. code-block:: c++

    extern "C" void launcher(int value,
                             const char* op_ptx, int op_ptx_size)
    {
      cudaSetDevice(0);

      // Kernel is now a string!
      std::string kernel_source = R"XXX(
        extern "C" __device__ int op(int a);

        extern "C" __global__ void kernel(int value) {
            printf("thread %d prints value %d\n", threadIdx.x, op(value));
        }
      )XXX";

Once that source string has been assembled, we compile it to PTX with
NVRTC:

.. code-block:: c++

      nvrtcProgram prog;
      const char *name = "test_kernel";
      nvrtcCreateProgram(&prog, kernel_source.c_str(), name, 0, nullptr, nullptr);

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);

      const int cc_major = deviceProp.major;
      const int cc_minor = deviceProp.minor;
      const std::string arch = std::string("-arch=sm_") + std::to_string(cc_major) + std::to_string(cc_minor);

      const char* args[] = { arch.c_str(), "-rdc=true" };
      const int num_args = sizeof(args) / sizeof(args[0]);

      // Compile the CUDA C++ kernel to PTX
      std::size_t ptx_size{};
      nvrtcResult compile_result = nvrtcCompileProgram(prog, num_args, args);
      nvrtcGetPTXSize(prog, &ptx_size);
      std::unique_ptr<char[]> ptx{new char[ptx_size]};
      nvrtcGetPTX(prog, ptx.get());
      nvrtcDestroyProgram(&prog);

At this point, we have two PTX inputs: PTX for the generated CUDA C++
kernel and PTX for the Python-defined operator. We can combine them
using nvJitLink:

.. code-block:: c++

      const char* link_options[] = { arch.c_str() };

      // Link PTX comping from kernel and PTX coming from Python operator
      nvJitLinkHandle handle;
      nvJitLinkCreate(&handle, 1, link_options);
      nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX, ptx.get(), ptx_size, name);
      nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX, op_ptx, op_ptx_size, name);
      nvJitLinkComplete(handle);

      // Get resulting cubin
      std::size_t cubin_size{};
      nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
      std::unique_ptr<char[]> cubin{new char[cubin_size]};
      nvJitLinkGetLinkedCubin(handle, cubin.get());
      nvJitLinkDestroy(&handle);

The result of linking is a cubin containing the generated kernel and
the Python-defined operator. We can load that cubin as a CUDA
library, retrieve the kernel from it, and launch it:

.. code-block:: c++

      // Load cubin
      CUlibrary library;
      cuLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);

      // Get kernel pointer out of the library
      CUkernel kernel;
      cuLibraryGetKernel(&kernel, library, "kernel");

      // Launch the kernel
      void *kernel_args[] = { &value };
      cuLaunchKernel((CUfunction)kernel, 1, 1, 1, 4, 1, 1, 0, 0, kernel_args, nullptr);

Now the output of the Python program would be:

.. code-block:: bash

    thread 0 prints value 84
    thread 1 prints value 84
    thread 2 prints value 84
    thread 3 prints value 84

This works, but it is still not optimal from a performance
perspective. If the operator were compiled as part of the same CUDA
C++ translation unit as the kernel, the compiler could inline it
directly. In the PTX-linked version above, however, the generated
cubin still contains a call to ``op`` instead of the operator body
itself.

To address this, we switch to a different intermediate representation.
Instead of PTX, we use `LTO-IR
<https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/>`_.
LTO-IR preserves enough information for link-time optimization, which
allows the operator to be inlined into the generated kernel.

On the Python side, switching from PTX to LTO-IR requires only a small
change:

.. code-block:: python

    ltoir, _ = numba.cuda.compile(op, sig=numba.int32(numba.int32), output="ltoir")

On the C++ side, we make the same switch from PTX to LTO-IR:

.. code-block:: c++

    const char* args[] = { arch.c_str(), "-rdc=true", "-dlto" };
    const int num_args = sizeof(args) / sizeof(args[0]);

    nvrtcResult compile_result = nvrtcCompileProgram(prog, num_args, args);

    std::size_t ltoir_size{};
    nvrtcGetLTOIRSize(prog, &ltoir_size);
    std::unique_ptr<char[]> ltoir{new char[ltoir_size]};
    nvrtcGetLTOIR(prog, ltoir.get());
    nvrtcDestroyProgram(&prog);

    const char* link_options[] = { "-lto", arch.c_str() };

    nvJitLinkHandle handle;
    nvJitLinkCreate(&handle, 2, link_options);
    nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, ltoir.get(), ltoir_size, name);
    nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, op_ltoir, op_ltoir_size, name);

If you inspect the generated cubin now, you will no longer see a call
to ``op``. Instead, the operator has been inlined into the kernel,
which improves performance. That is the key benefit of switching from
PTX to LTO-IR.

At this point, we have a working prototype that can pass
Python-defined operators into CUDA C++ kernels without sacrificing
performance. The next problem is user-defined data types. So far, the
examples have used built-in scalar types, but a practical API also
needs to support types whose layout is only known on the Python side.

Fortunately, the kernel source is already being assembled as a string at
runtime. That means we can also generate the type information needed by
the CUDA C++ side.

As a concrete example, suppose we want to pass a ``numba.complex128``
value into the kernel. The C++ side does not see the original Python
type definition, but that is not an issue. It only needs a storage
type with matching size and alignment, and can type-erase everything
else.

.. code-block:: c++

        extern "C" void launcher(void *value_ptr, int type_size, int type_alignment,
                                 const char* op_ltoir, int op_ltoir_size)
        {
            std::string storage_t = "struct __align__(" + std::to_string(type_alignment) + ")"
                                    + "storage_t { char data[" + std::to_string(type_size) + "]; };";

            std::string kernel_source = storage_t + R"XXX(
                extern "C" __device__ int op(char *state);

                extern "C" __global__ void kernel(storage_t value) {
                    printf("thread %d prints value %d\n", threadIdx.x, op(value.data));
                }
            )XXX";

            // ...
            void *kernel_args[] = { value_ptr };
            cuLaunchKernel((CUfunction)kernel, 1, 1, 1, 4, 1, 1, 0, 0, kernel_args, nullptr);

In this version, the operator takes a type-erased pointer. On the
Python side, we therefore pass a pointer to the ``numba.complex128``
value, together with the size and alignment needed to construct a
matching storage type on the C++ side:

.. code-block:: python

        import ctypes
        import numba
        import numba.cuda
        import numpy as np

        def op(value):
            return numba.int32(value[0].real + value[0].imag)

        value_type = numba.complex128
        context = numba.cuda.descriptor.cuda_target.target_context
        size = context.get_value_type(value_type).get_abi_size(context.target_data)
        alignment = context.get_value_type(value_type).get_abi_alignment(context.target_data)
        ltoir, _ = numba.cuda.compile(op, sig=numba.int32(numba.types.CPointer(value_type)), output='ltoir')

        value = np.array([1 + 2j], dtype=np.complex128)
        type_erased_value_ptr = value.ctypes.data_as(ctypes.c_void_p)

        bindings = ctypes.CDLL('./build/libkernel.so')
        bindings.launcher.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        bindings.launcher(type_erased_value_ptr, size, alignment, ltoir, len(ltoir))

In this example, we obtain the size and alignment of
``numba.complex128`` from Numba's type system. The remaining detail is
how to pass the value to ``cuLaunchKernel``. Kernel arguments are
described to ``cuLaunchKernel`` as pointers to host memory from which
the launch parameters are copied. In Python, that host-memory pointer
can be obtained in a few ways, for example with ``ctypes.byref`` or by
placing the value in a ``numpy.array`` and retrieving the array's
address with ``value.ctypes.data_as(ctypes.c_void_p)``.

One more ingredient is needed to get closer to the full
``cuda.compute`` implementation. The kernels in the CUDA C++ Core
Compute Libraries are templates, so our generated kernel must be a
template as well.

.. code-block:: c++

    std::string kernel_source = storage_t + R"XXX(
        extern "C" __device__ int op(char *state);

        template <class T>
        __global__ void kernel(T value) {
            printf("thread %d prints value %d\n", threadIdx.x, op(value.data));
        }
    )XXX";

Defining the kernel as a template is still not enough. We also need to
instantiate that template for the generated storage type. NVRTC
provides the necessary API for that:

.. code-block:: c++

    nvrtcProgram prog;
    const char *name = "test_kernel";
    nvrtcCreateProgram(&prog, kernel_source.c_str(), name, 0, nullptr, nullptr);

    // Get the name of the instantiated kernel
    std::string kernel_name = "kernel<storage_t>";

    // Instantiate kernel template
    nvrtcAddNameExpression(prog, kernel_name.c_str());
    // ...

    // Get lowered name of the kernel
    const char* kernel_lowered_name; // _Z6kernelI9storage_tEvT_
    nvrtcGetLoweredName(prog, kernel_name.c_str(), &kernel_lowered_name);
    // ...

    // Use it to get kernel pointer
    cuLibraryGetKernel(&kernel, library, kernel_lowered_name);

With these pieces in place, we can connect the simplified prototype back
to ``cuda.compute``.

At a high level, the ``cuda.compute`` API follows the same overall
structure, but packages it into three stages. Using parallel reduction
as an example:

#. In the first stage, ``cuda.compute.make_reduce_into(...)`` constructs
   a reusable reduction object:

   ``reducer = cuda.compute.make_reduce_into(d_in=d_in, d_out=d_out, op=op, h_init=h_init)``

   Here ``op`` is a Python function that must be made available to the
   CUDA kernel. As in the simplified prototype above, this stage
   compiles ``op`` to LTO-IR, generates the corresponding CUDA C++
   source, instantiates the necessary kernels, and compiles them with
   NVRTC. The resulting build state is stored inside the returned
   reduction object. At this stage, the concrete runtime values of the
   provided arrays do not matter yet; later calls may use different
   pointers or sizes, as long as the interface remains compatible.

#. In the second stage, that reduction object is used to query the
   amount of temporary storage required by the algorithm:

   ``temp_storage_size = reducer(temp_storage=None, d_in=d_input, d_out=d_output, num_items=num_items, op=op, h_init=h_init)``

   This returns the size of the temporary storage buffer, which must be
   allocated in device-accessible memory. No kernels are launched at
   this stage.

#. In the third stage, the algorithm is executed using the allocated
   temporary storage:

   ``reducer(temp_storage=temp_storage, d_in=d_input, d_out=d_output, num_items=num_items, op=op, h_init=h_init)``

   At this point, the kernels stored in the reduction object are
   launched and the reduction is performed.


For readers who want to connect this overview back to the source tree:

* The Python-facing API, operator compilation, and the logic for
  constructing and invoking reusable algorithm objects live under
  ``python/cuda_cccl/cuda/compute/``.
* The lower-level C/C++ runtime compilation and kernel-building
  machinery lives under ``c/parallel/``.
* User-facing examples for ``cuda.compute`` live under
  ``python/cuda_cccl/tests/compute/examples/``.
