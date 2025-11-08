#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
##########################
# RAPIDS Version Updater #
##########################

## Usage
# bash update_rapids_version.sh <new_version>

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_PATCH=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Need to distutils-normalize the versions for some use cases
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")

echo "Updating RAPIDS and devcontainers to $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Update CI files
sed_runner "/devcontainer_version/ s/'[0-9.]*'/'${NEXT_SHORT_TAG}'/g" ci/matrix.yaml
sed_runner "/devcontainer_version=/ s/=[0-9.]*/=${NEXT_SHORT_TAG}/g" ci/build_cuda_cccl_python.sh

function update_devcontainer() {
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${1}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${1}"
    sed_runner "s@\${localWorkspaceFolderBasename}-rapids-[0-9.]*@\${localWorkspaceFolderBasename}-rapids-${NEXT_SHORT_TAG}@g" "${1}"
}

# Update .devcontainer files
find .devcontainer/ ci/rapids/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    update_devcontainer "${filename}"
done
