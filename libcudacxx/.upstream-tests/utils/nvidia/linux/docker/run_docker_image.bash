#! /bin/bash

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

NVIDIAMODPROBE=$(which nvidia-modprobe)

# Arguments are a list of SM architectures to target; if there are no arguments,
# all known SM architectures are targeted.
COMPUTE_ARCHS_FLAG=""
if [ ! -z "${@}" ]
then
  COMPUTE_ARCHS_FLAG="-eLIBCUDACXX_COMPUTE_ARCHS=\"${@}\""
fi

JSON_OUTPUT_FLAG=""
DOCKER_VOLUMN_FLAG=""
if [ "${JSON_OUTPUT_PATH:-0}" != "0" ]
then
  JSON_OUTPUT_FLAG="-eJSON_OUTPUT_PATH=${JSON_OUTPUT_PATH}"
  DOCKER_VOLUMN_FLAG="-v ${SCRIPT_PATH}/mount:${JSON_OUTPUT_PATH}"
fi

# Ensure nvidia-uvm is loaded.
${NVIDIAMODPROBE} -u

# Support file interface with the host machine
mkdir -p ${SCRIPT_PATH}/mount
chmod 777 ${SCRIPT_PATH}/mount

docker run ${DOCKER_VOLUMN_FLAG} -t ${COMPUTE_ARCHS_FLAG} ${JSON_OUTPUT_FLAG} --privileged ${FINAL_IMAGE} 2>&1 \
  | while read l; do \
      echo "${LIBCUDACXX_DOCKER_OUTPUT_PREFIX}$(date --rfc-3339=seconds)| $l"; \
    done
if [ "${PIPESTATUS[0]}" != "0" ]; then exit 1; fi

