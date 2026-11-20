#!/usr/bin/env bash

# This script builds the CCCL Python documentation.
#
# Usage:
#   ./gen_python_docs.bash                      - Build documentation
#   ./gen_python_docs.bash --label 1.1.1        - Build as release 1.1.1 (default: unstable)
#   ./gen_python_docs.bash --allow-dep-install  - Build, installing Graphviz if it is missing
#   ./gen_python_docs.bash clean                - Clean build directory

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
cd "${SCRIPT_PATH}/python"

BUILDDIR="_build"
# This folder becomes /cccl/python/ in the published site.
HTML_DIR="${SCRIPT_PATH}/_build/artifacts/docs/python"

if [[ "${CLEAN}" = true ]]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILDDIR:?}" "${HTML_DIR}"
    exit 0
fi

VERSION="${LABEL:-${SPHINX_CCCL_VER:-unstable}}"
if [[ ! "${VERSION}" =~ ^(unstable|[0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    echo "Error: the version must be 'unstable' or MAJOR.MINOR.PATCH, got '${VERSION}'." >&2
    exit 1
fi
# conf.py reads the version from this variable.
export SPHINX_CCCL_VER="${VERSION}"
VERSIONED_HTML_DIR="${HTML_DIR}/${VERSION}"

# The cuda.coop docs draw diagrams with Graphviz
if ! command -v dot &>/dev/null; then
    if [[ "${ALLOW_DEP_INSTALL}" != true ]]; then
        echo "Error: Graphviz (dot) is missing. Install it, or re-run with --allow-dep-install." >&2
        exit 1
    fi
    echo "Installing Graphviz (--allow-dep-install)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq graphviz
fi

# Use the virtual environment shared with gen_docs.bash, creating it if needed
if [[ ! -d "${SCRIPT_PATH}/env" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "${SCRIPT_PATH}/env"
fi
# shellcheck disable=SC1091
source "${SCRIPT_PATH}/env/bin/activate"

if ! python -c "import sphinx" 2>/dev/null; then
    echo "Installing documentation dependencies..."
    python -m pip install -r "${SCRIPT_PATH}/requirements.txt"
fi

echo "Building Python documentation (version: ${VERSION})..."
rm -rf "${VERSIONED_HTML_DIR}"
mkdir -p "${VERSIONED_HTML_DIR}"
declare -a SPHINXOPTS="(${SPHINXOPTS:---keep-going -W})"
python -m sphinx.cmd.build -b html -d "${BUILDDIR}/doctrees" -j auto "." "${VERSIONED_HTML_DIR}" "${SPHINXOPTS[@]}"

# /cccl/python/ redirects to the unstable docs.
cp "${SCRIPT_PATH}/common/redirect.html" "${HTML_DIR}/index.html"

# Copy objects.inv to the root to support intersphinx consumers. Only the
# unstable build updates it.
if [[ "${VERSION}" == "unstable" && -f "${VERSIONED_HTML_DIR}/objects.inv" ]]; then
    cp "${VERSIONED_HTML_DIR}/objects.inv" "${HTML_DIR}/objects.inv"
fi

echo "Python documentation build complete: ${VERSIONED_HTML_DIR}"
