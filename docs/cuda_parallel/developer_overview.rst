CUDA Parallel Developer Overview
################################

This living document serves as a guide to the design of the internal structure of cuda.parallel.
On a high level, cuda.parallel provides Python bindings to CUDA C++ parallel algorithms.
These bindings combine CUDA Python compilation, CUDA C++ code generation, and CUDA C++ JIT compilation.
The binding process might be a bit hard to understand because it involves many moving parts.
To make it easier to understand, we'll start by covering a simplified version.
As we discover issues caused by this simplification,
we'll introduce workarounds along with references to the relevant source code.

Let's start with invoking a CUDA C++ kernel from Python.
Our simplified kernel accepts one integer and prints it.

.. code-block:: c++

    #include <cstdio>

    __global__ void kernel(int value) {
        std::printf("thread %d: %d\n", threadIdx.x, value);
    }

    extern "C" void launcher(int value) {
      kernel<<<1, 4>>>(value);
      cudaDeviceSynchronize();
    }


We can compile this code using nvcc:

.. code-block:: bash

    nvcc -Xcompiler=-fPIC -x cu kernel.cu -shared -o libkernel.so


We can now call a host function (``launcher``) from Python using ctypes:

.. code-block:: python

    import ctypes

    bindings = ctypes.CDLL('libkernel.so')
    bindings.launcher.argtypes = [ctypes.c_int]
    bindings.launcher(42)


This Python program will print the following output:

.. code-block:: bash

    thread 0: 42
    thread 1: 42
    thread 2: 42
    thread 3: 42


Let's say we are interested in computing parallel reduction.
If we only needed to support reduction of floating point numbers using the sum operation,
the code above would be enough.
However, we want to support reduction of any type using any operation.
Let's say, a user defines a Python function that we have to invoke on C++ end.
We can compile Python function to PTX using numba.cuda as follows:

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


On the C++ end, we could declare this function as extern one:


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


But how would we link the PTX code coming from Python with the CUDA C++ code?
We can't expect presense of nvcc on the user's machine.
That's one of the resons why we have to switch to using NVRTC compiler instead of nvcc.

NVRTC is a runtime compiler for CUDA C++.
NVRTC provides a C++ function that takes a string with CUDA C++ code and returns a machine code.
Let's change the launcher signature to accept a PTX string:


.. code-block:: python

    import ctypes
    import numba.cuda

    def op(value):
        return 2 * value

    ptx, _ = numba.cuda.compile(op, sig=numba.int32(numba.int32))

    bindings = ctypes.CDLL('./build/libkernel.so')
    bindings.launcher.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    bindings.launcher(42, ptx.encode('utf-8'), len(ptx))


The C++ signature of the launcher now would be:

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


We can compile the CUDA C++ code to PTX using NVRTC:


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


This gives us PTX code that we can link with the operator using nvJitLink:


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


Now we have linked codet that can be loaded as a CUDA library.
As soon as it's loaded, we can find the kernel in it.
As soon as we have the kernel, we can launch it:

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


This works, but it's not optimal.
If you take a look at the resulting cubin, you'll see that it contains a function call.
But if you'd compile this operator as part of the C++ translation unit, the function would be inlined.
Apart from presense of function call protocol, linking PTX causes extensive use of local memory.
Given millions of threads launched by parallel algorithms,
this leads to significant memory trafic and suboptimal performance.

To fix this, we have to use different intermediate representation.
So instead of PTX, we use `LTO-IR <https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/>`_.
It allows us to avoid consts associated with separate compilation.

As of the version 0.60, numba.cuda supports LTO-IR.
It's sufficient to change our compilation line to:

.. code-block:: python

    ltoir, _ = numba.cuda.compile(op, sig=numba.int32(numba.int32), target='cuda', options={'link': True})

On the C++ end, it's sufficient to replace PTX with LTOIR:


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


If you take a look at the generated cubin now, you'll see a single shuffle instruction instead of a function call.
In other words, LTO-IR allowed us to inline the operator and achieve better performance.

Now we have a working prototype allowing us to pass Python functions to CUDA C++ kernels withour sacrifising performance.
What remeains to be figured out is how we can support user-defined Python data types.
Fortunately, we already have our kernel as a string.
We can compose this string at runtime, adding the necessary type information.

As an example, let's try to pass a ``numba.complex128`` value into our kernel.
C++ part of our code doesn't see the definition of this type, but that's fine.
It's sufficient for us to create a storage structure with matching size and alignment and type-erase everything else.

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

The operator now takes a type-erased pointer.
Let's take a look at the Python side:

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


In the code above, we retreive type information (size and alignemnt) from the numba type system.
One of the problems is that ``cuLaunchKernel`` accepts an array of pointers to CPU memory,
from which it copies parameters for subsequent kernel launch.
To get the pointer to CPU memory, we could use ``ctypes.byref``.
Alternatively,
we could allocate memory using ``numpy.array`` and later retreive the pointer with ``.ctypes.data_as(ctypes.c_void_p)``.

The only missing part that separates us from cuda.parallel
is the fact that all the kernels in CUDA C++ Core Compute Libraries are templates.
Let's make our kernel a template as well.

.. code-block:: c++

    std::string kernel_source = storage_t + R"XXX(
        extern "C" __device__ int op(char *state);

        template <class T>
        __global__ void kernel(T value) {
            printf("thread %d prints value %d\n", threadIdx.x, op(value.data));
        }
    )XXX";


Unfortunately, this is not sufficient.
We have to instantiate the kernel template.
To do that, we can use the following NVRTC API:


.. code-block:: c++

    nvrtcProgram prog;
    const char *name = "test_kernel";
    nvrtcCreateProgram(&prog, kernel_source.c_str(), name, 0, nullptr, nullptr);

    // Get the name of the storage_t type
    std::string storage_t_name;
    nvrtcGetTypeName<storage_t>(&storage_t_name);

    // Get the name of the kernel
    std::string kernel_name = "kernel<" + storage_t_name + ">";

    // Instantiate kernel template
    nvrtcAddNameExpression(prog, kernel_name.c_str());
    // ...

    // Get lowered name of the kernel
    const char* kernel_lowered_name; // _Z6kernelI9storage_tEvT_
    nvrtcGetLoweredName(prog, kernel_name.c_str(), &kernel_lowered_name);
    // ...

    // Use it to get kernel pointer
    cuLibraryGetKernel(&kernel, library, kernel_lowered_name);


Let's see how the steps that we covered map to cuda.parallel.
On the high level, cuda.parallel API consists of three stages.
Let's take a look at these stages using the example of parallel reduction:

#. First step returns invocable: ``reduce_into = cudax.reduce_into(d_in, d_out, op, h_init)``.
   Here ``op`` is a Python function that we have to pass to the CUDA kernel.
   The ``cudax.reduce_into`` call starts by compiling ``op`` to LTO-IR, just like we did above.
   It then proceeds to the C++ part, which is responsible for composing a string with C++ code,
   instantiating kernels, and compiling them with NVRTC, just like we did above.
   Result of this compilation is stored inside ``reduce_info`` object.
   From the C++ perspective, runtime values of the provided parameters do not matter at this stage.
   In other words, concrete pointers or shapes of the provided containers can be different on subsequent stages.
#. Second step returns temporary storage needed for parallel algorithm:
   ``temp_storage_size = reduce_into(None, d_input, d_output, h_init)``.
   This storage has to be allocated in device-accessible memory.
   At this stage, no kernels are invoked.
#. Third step uses allocated temporary storage and kernels retreived from the cubin stored in the ``reduce_info`` object:
   ``reduce_into(temp_storage, d_input, d_output, h_init)``.
   This step launches the kernel and performs the reduction.
