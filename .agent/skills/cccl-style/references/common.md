# Common CCCL Style Guidance

Read the [CCCL C++ Coding Guidelines](https://nvidia.github.io/cccl/unstable/cccl/development/coding_guidelines.html),
which supersede everything below.
Then apply the guidance in this file across CCCL unless a path-specific style reference says otherwise.

## Variables

- All variables that are not modified must use `const`. This includes variables initialized by casts (`static_cast`, `reinterpret_cast`, `bit_cast`), function return values, and loop-invariant computations.
- Consider using plural names for array, span, list, e.g. `int values[4]` instead of `int value[4]`.
- Use uniform initialization for class constructors (not enforced to builtin types) and compile-time conversions, e.g. `constexpr auto x = int{sizeof(float)};`.

## Headers

- Use forward declaration, namely `__fwd/header.h` or direct type declaration, when possible instead of including the implementation header.
- Do not include headers in `cuda/std/__cccl/` directly; they are provided by `__config` or the prologue/epilogue mechanism.

## Functions

- Most functions with a non-void return type should use `[[nodiscard]]`; functions with known side effects may be exceptions.
- Functions that do not throw exceptions must use `noexcept`.
- Use `_CCCL_CONSTEVAL` when the function can only be evaluated at compile time.
- Use C++20 concept macros instead of SFINAE, e.g. `_CCCL_TEMPLATE(...)` and `_CCCL_REQUIRES(...)`.

## Function Calls And Types

- Static member functions of a class template inherit the class's namespace.

# Using CUDA APIs

- In headers, all CUDA Runtime (`cudaXxx(...)`) and CUDA Driver (`cuXxx(...)` or `cuda::__driver::xxxNoThrow(...)`) calls must have their return values handled or explicitly ignored with a comment why it's being ignored.
  - CUDA Runtime calls shall use `_CCCL_TRY_RUNTIME_API` or `_CCCL_ASSERT_RUNTIME_API` macros to handle the return values.
  - CUDA Driver calls shall use `_CCCL_TRY_DRIVER_API` or `_CCCL_ASSERT_DRIVER_API` macros to handle the return values.
  - Alternatively, the return value might be checked manually.
- In tests, all CUDA Runtime/Driver calls must have their return values checked either manually or using`assert(...)` in lit-style tests or `(CHECK|REQUIRE)_(CUDA|CUDART)` macros in catch2-style tests.

## Comments

- Commented code without a description is not allowed.

## General Guidelines

- Try to use modern C++ as much as possible. The repository supports C++17 but many more recent functionalities have been backported with functions and macros.

## Prevent Compiler Errors And Improve Compatibility

- Remove unused code, variables, functions, types, template parameters, headers, etc.
- Variables that are unsigned, or that can become unsigned after template instantiation, must not check for negative values directly. Use `cuda::std::is_unsigned_v<T> ? false : (var < 0)` instead.

## Compiler Compatibility

- Protect host-only code with `#if !_CCCL_COMPILER(NVRTC)`.

## Arch/SM-Scoped Perf Changes

Applies to any change that conditions a tuning constant, dispatch policy, or other perf-affecting
behavior like tuning encodings on compute capability or an SM arch macro.

- Never leave the comparison open-ended toward higher/future archs (e.g. `cc >= X` with no upper
  bound). An unbounded lower-only check silently applies to every arch above X, including ones
  that did not exist when the change was benchmarked.
- A closed range (`cc >= X && cc < Y`) is not automatically correct either: `Y` must sit exactly at
  the boundary of what was benchmarked, not at a rounder or more generous value picked out of
  convenience. A range wider than what was tested still leaks the change onto untested archs inside
  that range.
