# Doxygen + Breathe gotchas

## Suppressed warnings

`conf.py` suppresses two warning categories that cannot be fixed at the source level:

- `cpp.duplicate_declaration` — Breathe walks each Doxygen XML file independently. Symbols appearing in both a namespace XML and a class/group XML are emitted twice. No workaround on the CCCL side.
- `docutils` — Breathe renders complex C++ default arguments and SFINAE expressions as RST that docutils cannot parse (mismatched inline literals, unexpected braces). The C++ is valid; the RST Breathe emits is not.

## `_BREATHE_SKIP_SYMBOLS`

Defined in `docs/_ext/auto_api_generator.py`. API pages are not generated for these symbols to prevent unfixable build failures under `-W`:

| Symbol                                    | Reason                                                                                                                      |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `get_executor_func_t`                     | Function-pointer typedef; Breathe renders as `pos4(*)(pos4, dim4, dim4)` — Sphinx C++ parser rejects "Expected end of definition". |
| `cuda::experimental::stf::get_executor_func_t` | Same (qualified form).                                                                                                      |
| `partition_fn_t`                          | Same issue as `get_executor_func_t`.                                                                                         |
| `cuda::experimental::stf::partition_fn_t` | Same (qualified form).                                                                                                      |
| `property_with_value`                     | Variable template using `_CCCL_REQUIRES_EXPR` with `typename(...)` — Sphinx cannot parse the requires-expression expansion. |
| `has_property`                            | Variable template with `_CCCL_REQUIRES_EXPR` containing `const` — same parse failure.                                       |
| `cuda::experimental::group`               | Variable with complex type expression Sphinx C++ parser cannot handle.                                                       |

To add a new skip: append to `_BREATHE_SKIP_SYMBOLS` in `docs/_ext/auto_api_generator.py`. Both the qualified and unqualified forms may need to be listed (the generator uses different naming conventions per project).

## Exhale disabled

`exhale` is in `requirements.txt` but commented out in `conf.py`. It was disabled due to build timeouts. All API page generation goes through `auto_api_generator` instead.

## Doxygen version pinning

`gen_docs.bash` builds Doxygen 1.9.6 from source on first run and caches it under `docs/_build/doxygen-build/`. This pins the version for consistent output. If the cached binary is present, it is used unconditionally. To rebuild: `./docs/gen_docs.bash clean --all`.

## CUDA macro attributes

`conf.py` registers CCCL-specific macros as `cpp_id_attributes` so Breathe/Sphinx does not reject declarations annotated with `__device__`, `_CCCL_HOST_DEVICE`, `_CCCL_API`, and similar. If a new CCCL macro causes parse failures, add it to `cpp_id_attributes` or `cpp_paren_attributes` in `conf.py`.
