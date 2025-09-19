#!/usr/bin/env bash

# This script builds CCCL documentation using Sphinx directly
#
# Usage:
#   ./gen_docs.bash           - Build documentation
#   ./gen_docs.bash clean     - Clean build directory
#   ./gen_docs.bash clean --all - Clean build directory and Doxygen build
#
# The script will optionally build Doxygen 1.9.6 from source to ensure
# consistent documentation generation. The built Doxygen will be stored
# in _build/doxygen-build/ and reused for subsequent runs.

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
cd $SCRIPT_PATH

# Configuration
SPHINXOPTS="${SPHINXOPTS:---keep-going}"
BUILDDIR="_build"
DOXYGEN_BUILD_DIR="${SCRIPT_PATH}/_build/doxygen-build"
DOXYGEN_SRC_DIR="${SCRIPT_PATH}/_build/doxygen-src"
DOXYGEN_BIN="${DOXYGEN_BUILD_DIR}/bin/doxygen"

# Use custom-built doxygen if available, otherwise fall back to system doxygen
if [ -f "${DOXYGEN_BIN}" ]; then
    DOXYGEN="${DOXYGEN_BIN}"
else
    DOXYGEN="${DOXYGEN:-doxygen}"
fi

# Handle clean command
if [ "$1" = "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf ${BUILDDIR}/*
    if [ "$2" = "--all" ]; then
        echo "Also removing Doxygen source and build directories..."
        rm -rf "${DOXYGEN_SRC_DIR}" "${DOXYGEN_BUILD_DIR}"
    fi
    exit 0
fi

## Clean image directory, without this any artifacts will prevent fetching
rm -rf img
mkdir -p img

# Pull cub images
if [ ! -d cubimg ]; then
    git clone -b gh-pages https://github.com/NVlabs/cub.git cubimg
fi

if [ ! -n "$(find cubimg -name 'example_range.png')" ]; then
    wget -q https://raw.githubusercontent.com/NVIDIA/NVTX/release-v3/docs/images/example_range.png -O cubimg/example_range.png
fi

if [ ! -n "$(find img -name '*.png')" ]; then
    wget -q https://docs.nvidia.com/cuda/_static/Logo_and_CUDA.png -O img/logo.png

    # Parse files and collects unique names ending with .png
    imgs=( $(grep -R -o -h '[[:alpha:][:digit:]_]*.png' ../cub/cub | uniq) )
    imgs+=( "cub_overview.png" "nested_composition.png" "tile.png" "blocked.png" "striped.png" )

    for img in "${imgs[@]}"
    do
        echo ${img}
        cp cubimg/${img} img/${img}
    done
fi

# Function to build Doxygen 1.9.6
build_doxygen() {
    echo "Building Doxygen 1.9.6..."

    # Clone Doxygen if not already cloned
    if [ ! -d "${DOXYGEN_SRC_DIR}" ]; then
        echo "Cloning Doxygen repository..."
        git clone https://github.com/doxygen/doxygen.git "${DOXYGEN_SRC_DIR}"
    fi

    # Checkout Release_1_9_6
    cd "${DOXYGEN_SRC_DIR}"
    git fetch
    git checkout Release_1_9_6

    # Create build directory
    mkdir -p "${DOXYGEN_BUILD_DIR}"
    cd "${DOXYGEN_BUILD_DIR}"

    # Configure based on platform
    echo "Configuring Doxygen build..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Detected macOS, configuring with LLVM paths..."
        if ! command -v brew &> /dev/null; then
            echo "Warning: Homebrew not found, building without libclang support"
            cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
                "${DOXYGEN_SRC_DIR}"
        else
            cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
                -Duse_libclang=NO \
                -DBISON_EXECUTABLE="$(brew --prefix bison)/bin/bison" \
                "${DOXYGEN_SRC_DIR}"
        fi
    else
        # Linux/Ubuntu
        echo "Configuring for Linux/Ubuntu..."
        cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            -Duse_libclang=NO \
            "${DOXYGEN_SRC_DIR}"
    fi

    # Build Doxygen
    echo "Building Doxygen (this may take a few minutes)..."
    ninja

    echo "Doxygen 1.9.6 built successfully at ${DOXYGEN_BIN}"
    cd "${SCRIPT_PATH}"
}

# Check if custom Doxygen needs to be built
if [ ! -f "${DOXYGEN_BIN}" ]; then
    echo "Custom Doxygen 1.9.6 not found, building it now..."

    # Check for required build tools
    if ! command -v cmake &> /dev/null; then
        echo "Error: cmake is required to build Doxygen"
        echo "Please install cmake and try again"
        exit 1
    fi

    if ! command -v ninja &> /dev/null; then
        echo "Error: ninja is required to build Doxygen"
        echo "Please install ninja-build and try again"
        exit 1
    fi

    build_doxygen
    DOXYGEN="${DOXYGEN_BIN}"
else
    echo "Using custom-built Doxygen 1.9.6 from ${DOXYGEN_BIN}"
fi

# Check if documentation dependencies are installed
echo "Checking for documentation dependencies..."

# Use virtual environment if it exists, otherwise create one
if [ -d "env" ]; then
    echo "Using existing virtual environment..."
    source env/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv env
    source env/bin/activate
fi

# Check if dependencies are installed in the virtual environment
if ! python -c "import sphinx" 2>/dev/null; then
    echo "Installing documentation dependencies..."
    pip install -r requirements.txt || {
        echo "Error: Failed to install documentation dependencies"
        echo "Please install manually: pip install -r requirements.txt"
        exit 1
    }
fi

# Generate Doxygen XML in parallel (if doxygen is available)
if which ${DOXYGEN} > /dev/null 2>&1; then
    echo "Generating Doxygen XML..."
    mkdir -p ${BUILDDIR}/doxygen/cub ${BUILDDIR}/doxygen/thrust ${BUILDDIR}/doxygen/cudax ${BUILDDIR}/doxygen/libcudacxx

    # Copy all images to Doxygen XML output directories where they're expected
    for project in cub thrust cudax libcudacxx; do
        mkdir -p ${BUILDDIR}/doxygen/${project}/xml
        cp img/*.png ${BUILDDIR}/doxygen/${project}/xml/ 2>/dev/null || true
    done

    # Run all Doxygen builds in parallel
    (cd cub && ${DOXYGEN} Doxyfile) &
    (cd thrust && ${DOXYGEN} Doxyfile) &
    (cd cudax && ${DOXYGEN} Doxyfile) &
    (cd libcudacxx && ${DOXYGEN} Doxyfile) &
    wait

    echo "Doxygen complete"
else
    echo "Skipping Doxygen (not installed)"
fi

# Build Sphinx HTML documentation
echo "Building documentation with Sphinx..."
# Use the virtual environment's Python
python -m sphinx.cmd.build -b html -j auto . ${BUILDDIR}/html ${SPHINXOPTS}

echo "Documentation build complete! HTML output is in ${BUILDDIR}/html/"
