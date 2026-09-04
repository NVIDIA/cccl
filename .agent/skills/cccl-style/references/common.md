# Common CCCL Style Guidance

Read the [CCCL C++ Coding Guidelines](https://nvidia.github.io/cccl/unstable/cccl/development/coding_guidelines.html),
which supersede everything below.
Then apply the guidance in this file across CCCL unless a path-specific style reference says otherwise.

## Variables

- All variables that are not modified must use `const`. This includes variables initialized by casts (`static_cast`, `reinterpret_cast`, `bit_cast`), function return values, and loop-invariant computations.
- All variables that can be evaluated at compile-time must use `constexpr`.
- All `constexpr` variables at namespace/global scope must use `inline`, including variable templates.
- Consider using plural names for array, span, list, e.g. `int values[4]` instead of `int value[4]`.
- Use uniform initialization for class constructors (not enforced to builtin types) and compile-time conversions, e.g. `constexpr auto x = int{sizeof(float)};`.

## Headers

- Files must include all headers related to the symbols that they are using.
- Relying on transitive header inclusion is not allowed.
- Unneeded headers must be removed.
- All headers must have the correct license. This also applies to source files.
- All header inclusions must use the syntax `<header>`.
- Use forward declaration, namely `__fwd/header.h` or direct type declaration, when possible instead of including the implementation header.
- Headers should be the most precise available, e.g. `#include <cuda/std/__type_traits/is_array.h>`.
- Do not include headers in `cuda/std/__cccl/` directly; they are provided by `__config` or the prologue/epilogue mechanism.

## Functions

- Non-template, non-`constexpr` functions must use `inline`.
- Most functions with a non-void return type should use `[[nodiscard]]`; functions with known side effects may be exceptions.
- Functions that do not throw exceptions must use `noexcept`.
- Use `_CCCL_CONSTEVAL` when the function can only be evaluated at compile time.
- Use C++20 concept macros instead of SFINAE, e.g. `_CCCL_TEMPLATE(...)` and `_CCCL_REQUIRES(...)`.

## Function Calls And Types

- In headers, apply global qualification where the subproject requires it:
  - libcudacxx and cudax require free function calls to be fully qualified from the global namespace, e.g. `::cuda::ceil_div(...)`.
  - CUB applies this rule only to calls to symbols under the `::cuda` namespace hierarchy; otherwise follow existing CUB qualification style.
  - Thrust uses leading `::` for many symbols under the `::cuda` namespace hierarchy, but relies on ADL in many places and the blanket free-function qualification rule does not apply to those calls.
- For covered calls, this includes calls to functions defined in the same namespace, e.g. inside `cuda::`, call `::cuda::ceil_div(...)`, not `ceil_div(...)`. This does not apply to (static) member functions of classes. The only exceptions for covered calls are functions that are supposed to be found through argument-dependent lookup (ADL), such as `::cuda::std::swap` and `::cuda::std::get`. Those functions can be called unqualified with a preceding `using ::cuda::std::get;`.
- This global-qualification rule does not apply to source files such as tests and benchmarks.
- In headers, apply type-name qualification where the subproject requires it:
  - libcudacxx and cudax require type names to be fully qualified except when they are already declared in the current namespace or an enclosing one. Outside those namespaces, fully qualify `cuda::std` and standard integer type aliases such as `::cuda::std::size_t`.
  - CUB applies this rule only to type names under the `::cuda` namespace hierarchy. Do not apply the libcudacxx/cudax blanket type-qualification rule to CUB namespaces or `detail` namespaces.
  - Thrust does not apply the libcudacxx/cudax blanket type-qualification rule. It uses leading `::` for many `::cuda` and `::cuda::std` type names, but also uses Thrust namespace patterns; follow neighboring Thrust code.
- A local `using` declaration, e.g. `using ::cuda::std::size_t;`, is acceptable to avoid repetition within a function body.
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
