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

The exception carries the failing status as the API reported it, the *domain* that status belongs to (CUDA
Runtime, CUDA driver, or a library domain registered through ``cuda::cuda_status_domain``), and the source
location of the failure. ``status()`` returns the status seen as a CUDA Runtime error code: exact for a runtime
status, converted numerically for a driver status (the runtime reports driver-originated failures the same way),
and ``cudaErrorUnknown`` for any other domain. ``status<Status>()`` recovers the exact value of the type the
API produced.

.. code-block:: cpp

    class cuda_error : public std::runtime_error
    {
    public:
        cuda_error(cudaError_t status, const char* msg, const char* api = nullptr,
                   cuda::std::source_location loc = cuda::std::source_location::current());

        // CUresult, or any Status with a cuda_status_domain<Status> specialization
        template <class Status>
        cuda_error(Status status, const char* msg, const char* api = nullptr,
                   cuda::std::source_location loc = cuda::std::source_location::current());

        cudaError_t status() const noexcept;          // runtime view; cudaErrorUnknown for library domains
        template <class Status> bool holds() const noexcept;
        template <class Status> Status status() const noexcept;   // exact; precondition: holds<Status>()
        int raw_status() const noexcept;
        unsigned domain() const noexcept;
        const char* domain_name() const noexcept;
        const cuda::std::source_location& location() const noexcept;
    };

A library that reports failures through its own status enumeration registers it as a domain:

.. code-block:: cpp

    template <>
    struct cuda::cuda_status_domain<cublasStatus_t>
    {
        static constexpr unsigned id      = 2;        // unique among domains; 0 and 1 belong to libcu++
        static constexpr const char* name = "cuBLAS";
        static const char* description(cublasStatus_t s) { return cublasGetStatusString(s); }
    };

    throw cuda::cuda_error(status, "cublasGemmEx failed");   // what(): "file:line CUBLAS_STATUS_...(7): cublasGemmEx failed"
    catch (const cuda::cuda_error& e) { if (e.holds<cublasStatus_t>()) retry_with(e.status<cublasStatus_t>()); }
