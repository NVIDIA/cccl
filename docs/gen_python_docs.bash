#!/usr/bin/env bash

# Build the CCCL Python documentation as a standalone component.
#
# Usage:
#   ./gen_python_docs.bash                        - Build into artifacts/docs/python/unstable
#   ./gen_python_docs.bash --label 1.1.1          - Build into artifacts/docs/python/1.1.1
#   ./gen_python_docs.bash --allow-dep-install    - Build, installing missing deps
#   ./gen_python_docs.bash clean                  - Remove the Python build output
#
# The Python libraries ship on their own release line and are published as their
# own versioned site under /cccl/python/. This script produces that product's
# artifact and nothing else -- no Doxygen, no site assembly.
#
# Output mirrors gen_docs.bash: _build/artifacts/docs/python/<label>/ becomes
# /cccl/python/<label>/.

set -euo pipefail

ALLOW_DEP_INSTALL=false
CLEAN=false
LABEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-dep-install) ALLOW_DEP_INSTALL=true ;;
        clean)               CLEAN=true ;;
        --label)             LABEL="${2:-}"; shift ;;
        --label=*)           LABEL="${1#*=}" ;;
        *)                   echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

SCRIPT_PATH=$(cd "$(dirname "${0}")"; pwd -P)
cd "$SCRIPT_PATH"

BUILDDIR="_build"
HTML_DIR="${BUILDDIR}/artifacts/docs/python"

# Same rule as the C++ build: one value, and the directory name is the rendered
# stamp. --label takes precedence over the environment form.
VERSION="${LABEL:-${SPHINX_CCCL_VER:-unstable}}"

if [[ ! "${VERSION}" =~ ^(unstable|[0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    echo "Error: the version must be 'unstable' or an exact MAJOR.MINOR.PATCH release," >&2
    echo "       got '${VERSION}'." >&2
    echo "       Set it with --label, or with SPHINX_CCCL_VER for a local build." >&2
    exit 1
fi

export SPHINX_CCCL_VER="${VERSION}"

VERSIONED_HTML_DIR="${HTML_DIR}/${VERSION}"

if [[ "${CLEAN}" = true ]]; then
    echo "Removing ${HTML_DIR}..."
    rm -rf "${HTML_DIR:?}" "${BUILDDIR}/python-doctrees"
    exit 0
fi

if [[ -d "env" ]]; then
    # shellcheck disable=SC1091
    source env/bin/activate
elif [[ "${ALLOW_DEP_INSTALL}" = true ]]; then
    echo "Creating virtual environment..."
    python3 -m venv env
    # shellcheck disable=SC1091
    source env/bin/activate
else
    echo "Error: no virtual environment in ${SCRIPT_PATH}/env" >&2
    echo "       Re-run with --allow-dep-install, or create it yourself." >&2
    exit 1
fi

if ! python -c "import sphinx" 2>/dev/null; then
    if [[ "${ALLOW_DEP_INSTALL}" != true ]]; then
        echo "Error: documentation dependencies are not installed." >&2
        echo "       Re-run with --allow-dep-install, or: pip install -r requirements.txt" >&2
        exit 1
    fi
    echo "Installing documentation dependencies..."
    python3 -m pip install -r requirements.txt
fi

echo "Building Python documentation (version: ${VERSION})..."
rm -rf "${VERSIONED_HTML_DIR}"
mkdir -p "${VERSIONED_HTML_DIR}"

# Warnings are errors, so a cross-reference that stops resolving fails the build
# instead of leaving a dead link. Override with SPHINXOPTS, as in gen_docs.bash.
declare -a SPHINXOPTS="(${SPHINXOPTS:---keep-going -W})"

# The doctree cache is kept apart from the C++ one; the two builds share _build/.
python -m sphinx.cmd.build \
    -b html \
    -j auto \
    -d "${BUILDDIR}/python-doctrees" \
    "${SCRIPT_PATH}/python" \
    "${VERSIONED_HTML_DIR}" \
    "${SPHINXOPTS[@]}"

# Later products and the root project resolve against this inventory.
if [[ ! -f "${VERSIONED_HTML_DIR}/objects.inv" ]]; then
    echo "Error: the build produced no objects.inv." >&2
    echo "       Sphinx records every documented target there, so its absence" >&2
    echo "       means this artifact documents nothing." >&2
    exit 1
fi

# The product's landing redirect ships beside its version directory.
cp "${SCRIPT_PATH}/python_site/index.html" "${HTML_DIR}/index.html"

# Convenience inventory at the component root, tracking development docs.
if [[ "${VERSION}" == "unstable" && -f "${VERSIONED_HTML_DIR}/objects.inv" ]]; then
    cp "${VERSIONED_HTML_DIR}/objects.inv" "${HTML_DIR}/objects.inv"
fi

# The pages must claim the directory they are served from, or the switcher
# silently never highlights the current entry.
if ! grep -q "version_match = '${VERSION}'" "${VERSIONED_HTML_DIR}/index.html"; then
    echo "Error: pages are not stamped '${VERSION}'." >&2
    echo "       The switcher matches this stamp against its manifest entry." >&2
    exit 1
fi

# As in gen_docs.bash: the manifest lists this version plus every one already
# published, read from the documentation branch.
MANIFEST_ARGS=(--component python --version "${VERSION}" --out "${HTML_DIR}")
if [[ -n "${CCCL_DOCS_SITE_URL:-}" ]]; then
    MANIFEST_ARGS+=(--site-url "${CCCL_DOCS_SITE_URL}")
fi
if [[ -n "${CCCL_DOCS_BRANCH:-}" ]]; then
    MANIFEST_ARGS+=(--from-branch "${CCCL_DOCS_BRANCH}" --repo "${SCRIPT_PATH}/..")
fi
python3 "${SCRIPT_PATH}/make_manifest.py" "${MANIFEST_ARGS[@]}"

echo "Python documentation build complete: ${VERSIONED_HTML_DIR}"
