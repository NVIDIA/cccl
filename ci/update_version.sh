#!/bin/bash

# Usage: ./update_version.sh [--dry-run] <major> <minor> <patch>
# Example: ./update_version.sh --dry-run 2 2 1

# Run in root cccl/
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit

DRY_RUN=false

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; ;;
        *) break ;;
    esac
    shift
done

major="$1"
minor="$2"
patch="$3"
pymajor="0"
pyminor="1"

if [ -z "$major" ] || [ -z "$minor" ] || [ -z "$patch" ]; then
    echo "Usage: $0 [--dry-run] <major> <minor> <patch>"
    exit 1
fi

# Version file paths
CCCL_VERSION_FILE="libcudacxx/include/cuda/std/__cccl/version.h"
THRUST_VERSION_FILE="thrust/thrust/version.h"
CUB_VERSION_FILE="cub/cub/version.cuh"
CCCL_CMAKE_VERSION_FILE="lib/cmake/cccl/cccl-config-version.cmake"
CUB_CMAKE_VERSION_FILE="cub/cub/cmake/cub-config-version.cmake"
LIBCUDACXX_CMAKE_VERSION_FILE="libcudacxx/lib/cmake/libcudacxx/libcudacxx-config-version.cmake"
THRUST_CMAKE_VERSION_FILE="thrust/thrust/cmake/thrust-config-version.cmake"
CUDAX_CMAKE_VERSION_FILE="cudax/lib/cmake/cudax/cudax-config-version.cmake"
CUDA_COOPERATIVE_VERSION_FILE="python/cuda_cooperative/cuda/cooperative/_version.py"
CUDA_PARALLEL_VERSION_FILE="python/cuda_parallel/cuda/parallel/_version.py"

# Calculated version codes
new_cccl_version=$((major * 1000000 + minor * 1000 + patch))     # MMMmmmppp
new_thrust_cub_version=$((major * 100000 + minor * 100 + patch)) # MMMmmmpp

# Fetch current version from file
current_cccl_version=$(grep -oP "define CCCL_VERSION \K[0-9]+" "$CCCL_VERSION_FILE")

# Fetch the latest tag from git and strip the 'v' prefix if present
latest_tag=$(git tag --sort=-v:refname | head -n 1 | sed 's/^v//')

# Since the tags and versions are numerically comparable, we cast them to integers
latest_tag_version=$(echo "$latest_tag" | awk -F. '{ printf("%d%03d%03d", $1,$2,$3) }')

echo "Running in $(pwd)"
echo "New MMMmmmppp version: $new_cccl_version"
echo "New MMMmmmpp  version: $new_thrust_cub_version"
echo "Current CCCL version:  $current_cccl_version"
echo "Latest git tag:        $latest_tag"

# Check if new version is less than or equal to current or the latest tag
if (( new_cccl_version < current_cccl_version )) || (( new_cccl_version < latest_tag_version )); then
    echo "Error: New version $new_cccl_version is less than current version $current_cccl_version or latest git tag version $latest_tag_version."
    exit 1
fi

update_file () {
    local file=$1
    local pattern=$2
    local new_value=$3
    if [ "$DRY_RUN" = true ]; then
        local temp_file=$(mktemp)
        sed "s/$pattern/$new_value/g" "$file" > "$temp_file"
        diff --color=auto -U 0 "$file" "$temp_file" || true
        rm "$temp_file"
    else
        sed -i "s/$pattern/$new_value/" "$file"
    fi
}

# Update version information in files
update_file "$CCCL_VERSION_FILE" "^#define CCCL_VERSION \([0-9]\+\)" "#define CCCL_VERSION $new_cccl_version"
update_file "$THRUST_VERSION_FILE" "^#define THRUST_VERSION \([0-9]\+\)" "#define THRUST_VERSION $new_thrust_cub_version"
update_file "$CUB_VERSION_FILE" "^#define CUB_VERSION \([0-9]\+\)" "#define CUB_VERSION $new_thrust_cub_version"

update_file "$CUB_CMAKE_VERSION_FILE" "set(CUB_VERSION_MAJOR \([0-9]\+\))" "set(CUB_VERSION_MAJOR $major)"
update_file "$CUB_CMAKE_VERSION_FILE" "set(CUB_VERSION_MINOR \([0-9]\+\))" "set(CUB_VERSION_MINOR $minor)"
update_file "$CUB_CMAKE_VERSION_FILE" "set(CUB_VERSION_PATCH \([0-9]\+\))" "set(CUB_VERSION_PATCH $patch)"

update_file "$LIBCUDACXX_CMAKE_VERSION_FILE" "set(libcudacxx_VERSION_MAJOR \([0-9]\+\))" "set(libcudacxx_VERSION_MAJOR $major)"
update_file "$LIBCUDACXX_CMAKE_VERSION_FILE" "set(libcudacxx_VERSION_MINOR \([0-9]\+\))" "set(libcudacxx_VERSION_MINOR $minor)"
update_file "$LIBCUDACXX_CMAKE_VERSION_FILE" "set(libcudacxx_VERSION_PATCH \([0-9]\+\))" "set(libcudacxx_VERSION_PATCH $patch)"

update_file "$THRUST_CMAKE_VERSION_FILE" "set(THRUST_VERSION_MAJOR \([0-9]\+\))" "set(THRUST_VERSION_MAJOR $major)"
update_file "$THRUST_CMAKE_VERSION_FILE" "set(THRUST_VERSION_MINOR \([0-9]\+\))" "set(THRUST_VERSION_MINOR $minor)"
update_file "$THRUST_CMAKE_VERSION_FILE" "set(THRUST_VERSION_PATCH \([0-9]\+\))" "set(THRUST_VERSION_PATCH $patch)"

update_file "$CCCL_CMAKE_VERSION_FILE" "set(CCCL_VERSION_MAJOR \([0-9]\+\))" "set(CCCL_VERSION_MAJOR $major)"
update_file "$CCCL_CMAKE_VERSION_FILE" "set(CCCL_VERSION_MINOR \([0-9]\+\))" "set(CCCL_VERSION_MINOR $minor)"
update_file "$CCCL_CMAKE_VERSION_FILE" "set(CCCL_VERSION_PATCH \([0-9]\+\))" "set(CCCL_VERSION_PATCH $patch)"

update_file "$CUDAX_CMAKE_VERSION_FILE" "set(cudax_VERSION_MAJOR \([0-9]\+\))" "set(cudax_VERSION_MAJOR $major)"
update_file "$CUDAX_CMAKE_VERSION_FILE" "set(cudax_VERSION_MINOR \([0-9]\+\))" "set(cudax_VERSION_MINOR $minor)"
update_file "$CUDAX_CMAKE_VERSION_FILE" "set(cudax_VERSION_PATCH \([0-9]\+\))" "set(cudax_VERSION_PATCH $patch)"

update_file "$CUDA_COOPERATIVE_VERSION_FILE" "^__version__ = \"\([0-9.]\+\)\"" "__version__ = \"$pymajor.$pyminor.$major.$minor.$patch\""
update_file "$CUDA_PARALLEL_VERSION_FILE" "^__version__ = \"\([0-9.]\+\)\"" "__version__ = \"$pymajor.$pyminor.$major.$minor.$patch\""

if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed. No changes made."
else
    echo "Version updated to $major.$minor.$patch"
fi
