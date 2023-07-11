#! /bin/bash

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

TMP_PATH=$(mktemp -d --suffix=-${FINAL_NAME})
NVIDIAMODPROBE=$(which nvidia-modprobe)
LIBCUDA=$(ldconfig -p | grep libcuda.so | tr ' ' '\n' | grep / | head -n 1 | tr '\n' ' ' | sed 's/ *$//')
LIBNVIDIAPTXJITCOMPILER=$(ldconfig -p | grep libnvidia-ptxjitcompiler.so | tr ' ' '\n' | grep / | head -n 1 | tr '\n' ' ' | sed 's/ *$//')

echo "CUDA driver utilities found:"

for utility in ${NVIDIAMODPROBE}
do
  echo "  ${utility}"
done

echo "CUDA driver libraries found:"

for library in ${LIBCUDA} ${LIBNVIDIAPTXJITCOMPILER}
do
  echo "  ${library}"
done

# Ensure nvidia-uvm is loaded.
${NVIDIAMODPROBE} -u

cp ${NVIDIAMODPROBE} ${LIBCUDA} ${LIBNVIDIAPTXJITCOMPILER} ${TMP_PATH}
if [ "${?}" != "0" ]; then exit 1; fi

chmod -R 755 ${TMP_PATH}
if [ "${?}" != "0" ]; then exit 1; fi

docker -D build -t ${FINAL_IMAGE} -f ${FINAL_DOCKERFILE} ${TMP_PATH} 2>&1 \
  | while read l; do \
      echo "${LIBCUDACXX_DOCKER_OUTPUT_PREFIX}$(date --rfc-3339=seconds)| $l"; \
    done
if [ "${PIPESTATUS[0]}" != "0" ]; then exit 1; fi

rm ${TMP_PATH}/*
rmdir ${TMP_PATH}

