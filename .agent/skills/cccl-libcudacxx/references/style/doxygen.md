# Comments and Doxygen

Conventions for inline comments and Doxygen documentation in libcudacxx production headers.

## Inline comments

- Do not leave commented-out code without an explanatory comment describing why it is
  retained.
- Use `//` for inline implementation notes. Use `//!` (Doxygen) for documentation that is
  meant to surface in the rendered API reference.

## Doxygen on public API functions

Functions documented with Doxygen must carry all three of `@brief`, `@param`, and `@return`;
partial documentation is not allowed.

```cpp
//! @brief One-line description of what the function does.
//! @param[in] x          Description of x.
//! @param[out] result    Description of result.
//! @return What the return value means.
```

- `@param` directions are `[in]`, `[out]`, or `[in,out]`.
- Omit `//! @return` only for `void` functions.
- The description text must reflect current behavior; stale Doxygen is treated as a bug and
  fixed in the same patch as the underlying behavior change.

## When Doxygen is required

Apply Doxygen to public API functions — anything documented in the public reference. Apply
it to internal helpers only when the helper's contract is non-obvious to a future reader.
