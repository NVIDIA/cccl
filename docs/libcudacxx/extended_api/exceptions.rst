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

The exception carries the failing status as the API reported it, the type that status came from, and the source
location of the failure. Any status enumeration can be thrown: ``cudaError_t``, ``CUresult``, or the status type
of any other CUDA library. ``status()`` returns the status seen as a CUDA Runtime error code: exact for a
``cudaError_t``, converted numerically for a ``CUresult`` (the runtime reports driver-originated failures the same
way), and ``cudaErrorUnknown`` for any other type. ``status<Status>()`` recovers the exact value.

.. code-block:: cpp

    class cuda_error : public std::runtime_error
    {
    public:
        template <class Status>   // any status enumeration
        cuda_error(Status status, const char* msg, const char* api = nullptr,
                   cuda::std::source_location loc = cuda::std::source_location::current());

        cudaError_t status() const noexcept;                      // runtime view
        template <class Status> bool holds() const noexcept;      // did the status come from a Status?
        template <class Status> Status status() const noexcept;   // the status object itself; precondition: holds<Status>()
        cuda::std::int64_t raw_code() const noexcept;             // the code as reported
        cuda::std::string_view status_type_name() const noexcept; // e.g. "CUresult"
        const cuda::std::source_location& location() const noexcept;
    };

``what()`` reads ``file:line api status_type(code): text: msg``, where ``text`` is the library's description of
the code when one is known.

Every enumeration is accepted without registration: zero is success and the value is the code. Which other
status types are accepted, and how a library supplies text for its codes, is an implementation detail for now.

.. code-block:: cpp

    try
    {
        throw cuda::cuda_error(cufftPlan1d(&plan, n, CUFFT_C2C, 1), "cufftPlan1d failed");
    }
    catch (const cuda::cuda_error& e)
    {
        if (e.holds<cufftResult>())
        {
            cufftResult r = e.status<cufftResult>(); // the exact value
        }
    }
