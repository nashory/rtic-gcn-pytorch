#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Run this script at project root by "./dev/linter.sh" before you commit

{
  V=$(black --version|cut '-d ' -f3)
  code='import distutils.version; assert "19.3" < distutils.version.LooseVersion("'$V'")'
  python -c "${code}" 2> /dev/null
} || {
  echo "Linter requires black 19.3b0 or higher!"
  exit 1
}

DIR=$(pwd)
echo "$DIR"

echo "Running isort..."
isort --sp "${DIR}" "${DIR}"

echo "Running black..."
black "${DIR}"
