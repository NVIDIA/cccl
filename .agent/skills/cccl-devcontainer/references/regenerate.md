# Regenerating devcontainer subdirs

## When to regenerate

The per-combination subdirs under `.devcontainer/` (e.g. `.devcontainer/cuda13.2-gcc14/`) and
their `devcontainer.json` files are **generated**. Direct edits to them are overwritten on the
next regeneration run. Regenerate when:

- Adding or removing a CUDA × host-compiler combination from `ci/matrix.yaml`.
- Changing the base `.devcontainer/devcontainer.json` template.
- Pruning stale subdirs that no longer match the matrix.

## How to regenerate

1. Edit `ci/matrix.yaml` — the `dc` entries (and `dc_ext` for extended-CTK images) control which
   CUDA × host-compiler combinations exist.
2. If the template itself needs changing, edit `.devcontainer/devcontainer.json` (the base template,
   not a per-combination subdir).
3. From the repo root:

```
.devcontainer/make_devcontainers.sh --clean
```

`--clean` prunes subdirs for combinations that no longer appear in the matrix.

## What gets rewritten

- `.devcontainer/cuda<version>-<host>/devcontainer.json` — one file per matrix combination.
- Stale subdirs are removed when `--clean` is passed.
- The base `.devcontainer/devcontainer.json` template is **not** modified by the script.

## Validating after regeneration

Push the regenerated files. CI's "Validate Devcontainer" jobs run automatically and confirm
each generated `devcontainer.json` is well-formed.

`[skip-vdc]` blocks those jobs. Do not use it on PRs that modify `.devcontainer/`, `ci/`, or
`.github/`.
