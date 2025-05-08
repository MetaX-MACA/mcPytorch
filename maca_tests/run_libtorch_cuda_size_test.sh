#!/bin/bash
# get code branch, commit_id, and libtorch_cuda.so size(byte)
cd $(dirname $0)
pushd ..
git config --global --add safe.directory $(pwd)
# branch_name=$(git rev-parse --abbrev-ref --symbolic-full-name @{u})
branch_name='origin/dev_2.0'
version_file='version.txt'
if grep -q "2.1." "$version_file"; then
  branch_name='origin/dev_2.1'
fi
if grep -q "2.4." "$version_file"; then
  branch_name='origin/dev_2.4'
fi

current_commit_id=$(git log -1 --format=%H)
benchmark_commit_id=$(git log -1 --skip=1 --format=%H)
if [ -z "$CONDA_PREFIX" ]; then
  echo "No Conda environment is currently activated."
  exit 1
fi
libtorch_cuda_so=$(stat --format="%s" $(find ${CONDA_PREFIX} -name "libtorch_cuda.so"))
popd

# run test
python test_libtorch_cuda_size.py -branch $branch_name -benchmark_commit_id ${benchmark_commit_id} -current_commit_id ${current_commit_id} -libtorch_cuda_so ${libtorch_cuda_so}

if [[ $? != 0 ]]; then
    echo "test_libtorch_cuda_size.py failed!"
    exit 1
fi
exit 0
