#!/bin/bash
set -x

echo "MACA_VERSION: "${MACA_VERSION}

cur_dir="$(cd $(dirname $0);pwd)/"
cd ${cur_dir}

if [[ -e /.dockerenv ]]; then
    mkdir -p /tmp/$USER
    export HOME=/tmp/$USER
    ./build_and_run_impl.sh "$@"
else
    flock -x -w 10800 ~/maca_build_cubridge.lock ./build_and_run_impl.sh "$@"
fi

exit $?
