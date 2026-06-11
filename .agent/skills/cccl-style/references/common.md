# Common CCCL Style Guidance

Apply this guidance across CCCL unless a path-specific style reference says otherwise.

## Naming Style

- Macros: macro style, e.g. `MY_MACRO`.

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

## Function Calls And Types

- In headers, free function calls must be fully qualified from the global namespace, e.g. `::cuda::ceil_div(...)`.
- This global-qualification rule does not apply to source files such as tests and benchmarks.
- Type names must be fully qualified except when they are already declared in the current namespace or an enclosing one.
- Outside those namespaces, fully qualify `cuda::std` and standard integer type aliases such as `::cuda::std::size_t`.
- Static member functions of a class template inherit the class's namespace.

## Comments

- Commented code without a description is not allowed.

## General Guidelines

- The code must reuse `cuda/` or `cuda/std` functionalities as much as possible, including macros.
- Try to use modern C++ as much as possible. The repository supports C++17 but many more recent functionalities have been backported with functions and macros.

## Prevent Compiler Errors And Improve Compatibility

- Remove unused code, variables, functions, types, template parameters, headers, etc.
- Variables that are unsigned, or that can become unsigned after template instantiation, must not check for negative values directly. Use `cuda::std::is_unsigned_v<T> ? false : (var < 0)` instead.
