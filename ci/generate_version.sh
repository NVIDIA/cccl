#!/bin/bash

# Generate a version number string using metadata from git or JSON.
# Use the PyPi package versioning convention for pre-release or
# post release patches.

# Set some defaults for variables.
CCCL_BRANCH="${CCCL_BRANCH:-dev}"
PACKAGE_VERSION_PREFIX="${PACKAGE_VERSION_PREFIX:-}"

GIT_DESCRIBE_TAG=$(git describe --abbrev=0)
GIT_DESCRIBE_NUMBER=$(git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count)

JSON_VERSION=$(jq -r .full /workspace/cccl-version.json)

# Generate a suffix depending on release or dev branch.
PACKAGE_VERSION_SUFFIX=""
if [[ "${GIT_DESCRIBE_NUMBER}" != "0" ]]; then
  if [[ ${CCCL_BRANCH} == "dev" ]]; then
    PACKAGE_VERSION_SUFFIX=".dev${GIT_DESCRIBE_NUMBER}"
  else
    PACKAGE_VERSION_SUFFIX=".post${GIT_DESCRIBE_NUMBER}"
  fi
fi

# If using Git metadata this could generate it from the last tag.
# VERSION="${GIT_DESCRIBE_TAG#v}.dev${GIT_DESCRIBE_NUMBER}"

# Generate the version using a combination of JSON and git commit number.
VERSION="${PACKAGE_VERSION_PREFIX}${JSON_VERSION}${PACKAGE_VERSION_SUFFIX}"

echo -n "${VERSION}"
