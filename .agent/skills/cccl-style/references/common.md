# Common CCCL Style Guidance

Apply this guidance across CCCL unless a path-specific style reference says otherwise.

## Naming Style

- Macros: macro style, e.g. `MY_MACRO`.
- Template parameters: CamelCase, e.g. `MyParameter`.
- All other symbols: snake style, e.g. `my_variable`.


## Variables

- All variables that are not modified must use `const`. This includes variables initialized by casts (`static_cast`, `reinterpret_cast`, `bit_cast`), function return values, and loop-invariant computations.
- All variables that can be evaluated at compile-time must use `constexpr`.
- All `constexpr` variables at namespace/global scope must use `inline`, including `template` variables.
- Consider using plural names for array, span, list, e.g. `int values[4]` instead of `int value[4]`.
- Use uniform initialization for class constructors (not enforced to builtin types) and compile-time conversions, e.g. `constexpr auto x = int{sizeof(float)};`.

## Headers

- Files must include all headers related to the symbols that they are using.
- No transitive header inclusion are allowed.
- Unneeded headers must be removed.
- All headers must have the correct license.
- All header inclusions must use the syntax `<header>`.
- Use forward declaration, namely `__fwd/header.h` or direct type declaration, when possible instead of including the implementation header.
- Headers should be the most precise available, e.g. `#include <cuda/std/__type_traits/is_array.h>`.
- Do not include headers in `cuda/std/__cccl/` directly; they are provided by `__config` or the prologue/epilogue mechanism.

## Functions

- Functions must be marked `_CCCL_HOST_API`, `_CCCL_DEVICE_API`, or `_CCCL_API`.
- Non-template, non-`constexpr` functions must use `inline`.
- Most functions with a non-void return type should use `[[nodiscard]]`; functions with known side effects may be exceptions.
- Functions that do not throw exceptions must use `noexcept`.
- Use `_CCCL_CONSTEVAL` when the function can only be evaluated at compile time.
- Use C++20 concept macros instead of SFINAE, e.g. `_CCCL_TEMPLATE(...)` and `_CCCL_REQUIRES(...)`.

## Function Calls And Types

- In headers, free function calls must be fully qualified from the global namespace, e.g. `::cuda::ceil_div(...)`. This includes calls to functions defined in the same namespace, e.g. inside `cuda::`, call `::cuda::ceil_div(...)`, not `ceil_div(...)`. This does not apply to (static) member functions of classes. The only exceptions to this rule are functions that are supposed to be found through argument-dependent lookup (ADL), such as `::cuda::std::swap` and `::cuda::std::get`. Those functions can be called unqualified with a preceding `using ::cuda::std::get;`.
- This global-qualification rule does not apply to source files such as tests and benchmarks.
- Type names must be fully qualified except when they are already declared in the current namespace or an enclosing one.
- Outside those namespaces, fully qualify `cuda::std` and standard integer type aliases such as `::cuda::std::size_t`. A local `using` declaration, e.g. `using ::cuda::std::size_t;`, is acceptable to avoid repetition within a function body.
- Static member functions of a class template inherit the class's namespace.

## Comments

- Commented code without a description is not allowed.

## General Guidelines

- The code must reuse `cuda/` or `cuda/std` functionalities as much as possible, including macros.
- Try to use modern C++ as much as possible. The repository supports C++17 but many more recent functionalities have been backported with functions and macros.

## Prevent Compiler Errors And Improve Compatibility

- Remove unused code, variables, functions, types, template parameters, headers, etc.
- Variables that are unsigned, or that can become unsigned after template instantiation, must not check for negative values directly. Use `cuda::std::is_unsigned_v<T> ? false : (var < 0)` instead.

## Compiler Compatibility

- Protect host-only code with `#if !_CCCL_COMPILER(NVRTC)`.
