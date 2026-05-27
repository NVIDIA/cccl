---
description: |
  CCCL's documentation system — Sphinx pages + Doxygen API extraction + per-library subtrees.
  Covers the docs/ layout, build script, deploy workflow, and Breathe integration.
  Triggers: "how do I build the docs", "how do docs deploy", "where are the docs sources",
  "sphinx docs", "doxygen api docs".
---

# cccl-docs

CCCL's documentation is a Sphinx site with Doxygen-generated API references bridged via Breathe. The source lives entirely under `docs/`. There is no CMake-based doc target — the build entry point is a single shell script.

## Layout

```
docs/
├── conf.py                 ← Sphinx configuration (extensions, Breathe projects, theme)
├── requirements.txt        ← Python deps (sphinx, breathe, myst-parser, nvidia-sphinx-theme, …)
├── gen_docs.bash           ← build entry point
├── scrape_docs.bash        ← post-build page-list scraper
├── index.rst               ← site root (cpp, python, maintainers)
├── cpp.rst                 ← C++ library landing page
├── cccl/                   ← cross-library guides (contributing, migration, macros, dev/)
├── libcudacxx/             ← libcudacxx docs + Doxyfile
├── cub/                    ← CUB docs + Doxyfile
├── thrust/                 ← Thrust docs + Doxyfile
├── cudax/                  ← cudax docs + Doxyfile
├── python/                 ← Python package docs (compute, coop)
├── maintainers/            ← branching, backport, coderabbit guides
├── _ext/auto_api_generator.py  ← custom Sphinx extension (generates API pages from Doxygen XML)
└── _build/                 ← generated output (gitignored)
```

Each C++ library (`cub`, `thrust`, `libcudacxx`, `cudax`) has its own `Doxyfile` that extracts XML into `docs/_build/doxygen/<lib>/xml/`. Breathe reads that XML; `auto_api_generator.py` drives page generation.

## Building locally

Run from the repo root (Linux only; needs `cmake`, `ninja`, `flex`, `bison`, `python3-venv`):

```
./docs/gen_docs.bash
```

Pass `--allow-dep-install` to auto-install missing system packages via `apt`. The script:

1. Builds Doxygen 1.9.6 from source on first run (cached under `docs/_build/doxygen-build/`).
2. Creates a Python venv at `docs/env/` and installs `docs/requirements.txt`.
3. Runs each library's Doxygen build in parallel.
4. Runs `sphinx.cmd.build -b html`.
5. Reorganises output into `docs/_build/html/<VERSION>/` and writes `nv-versions.json`.

To build inside a container, route to `cccl-devcontainer` first. The custom Doxygen build takes several minutes on first run; subsequent runs reuse the cached binary.

Clean the build output:

```
./docs/gen_docs.bash clean        # removes _build/html
./docs/gen_docs.bash clean --all  # also removes cached Doxygen source + binary
```

## Deploy workflow

`.github/workflows/docs-deploy.yml` — triggers on push to `main` and `workflow_dispatch`.

- `main` → publishes under `docs/unstable/` with `is_latest=true`.
- `branch/X.Y` → publishes under `docs/X.Y/`.
- `destination_override` input overrides the target directory (useful for fork testing).

The workflow delegates to `.github/actions/docs-build/action.yml`, which calls `gen_docs.bash --allow-dep-install` and copies `docs/_build/html/*` to `_site/`. The deploy step (`peaceiris/actions-gh-pages`) pushes to the `gh-pages` branch with `keep_files: true`.

Published site: `https://nvidia.github.io/cccl/`.

## Sphinx + Doxygen integration

Breathe is the bridge. `conf.py` declares four Breathe projects (`cub`, `thrust`, `libcudacxx`, `cudax`), each pointing at its Doxygen XML directory. The custom extension `_ext/auto_api_generator.py` walks the XML and generates individual `.rst` pages per API symbol, skipping a small set of symbols that Breathe cannot parse (`_BREATHE_SKIP_SYMBOLS`).

Exhale is listed in `requirements.txt` but disabled in `conf.py` (build timeouts). All API page generation goes through `auto_api_generator`.

Theme: `nvidia-sphinx-theme`. Version switcher JSON: `nv-versions.json` at the site root.

## Per-library API doc structure

| Library    | Doxyfile                      | Input headers                                            |
|------------|-------------------------------|----------------------------------------------------------|
| CUB        | `docs/cub/Doxyfile`           | `cub/cub/**` (excludes `detail/`, `dispatch/`, `kernels/`) |
| Thrust     | `docs/thrust/Doxyfile`        | Thrust public headers                                   |
| libcudacxx | `docs/libcudacxx/Doxyfile`    | libcudacxx public headers                               |
| cudax      | `docs/cudax/Doxyfile`         | cudax public headers                                    |

Python API docs use Sphinx `autodoc` + `autosummary` (no Doxygen). Source is `python/cuda_cccl/`.

## Additional resources

- `references/doxygen-breathe-gotchas.md` — known parse failures, `_BREATHE_SKIP_SYMBOLS`, suppressed warnings.
