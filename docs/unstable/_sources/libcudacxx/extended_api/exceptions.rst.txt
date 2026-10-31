.. _libcudacxx-extended-api-exceptions:

Exception Handling
==================

Standard C++ exception handling (``try``, ``catch``, ``throw``) is not supported in CUDA device code, while it is enabled by default in host code.

**Device code**

``libcu++`` maps exceptions to ``cuda::std::terminate()`` calls in device code, which translates to ``__trap()`` and terminates the kernel.

**Host code**

``libcu++`` allows users to manually disable exceptions in host code in two ways:

- By defining ``CCCL_DISABLE_EXCEPTIONS`` before including any library headers.
- By compiling with ``-fno-exceptions`` compiler flag with ``gcc`` or ``clang``, or ``/EH-`` compiler flag with ``msvc``.

If exceptions are disabled, a ``throw`` exception is translated into a `cuda::std::terminate() <https://en.cppreference.com/w/cpp/error/terminate.html>`__ call, which terminates the program.

``cuda::cuda_error``
--------------------

Exception class thrown when a CUDA error is encountered. It inherits from ``std::runtime_error``.

.. code-block:: cpp

    class cuda_error : public std::runtime_error
    {
    public:
        cuda_error(cudaError_t status, const char* msg);

        cudaError_t status() const noexcept;
    };
