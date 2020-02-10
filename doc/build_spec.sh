#!/bin/bash
#
# Copyright 2016-2020 Intel Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SPEC_DIR=$SCRIPT_DIR/spec
SPEC_SOURCE_DIR=$SPEC_DIR/source
SPEC_RST_BUILD_DIR=$SPEC_DIR/build
SPEC_RST_HTML_BUILD_DIR=$SPEC_RST_BUILD_DIR/html
RST_DIR=$SCRIPT_DIR/rst
RST_SOURCE_DIR=$RST_DIR/source

if ! [ -x "$(command -v sphinx-build)" ]
then
  echo 'Error: Sphinx not found.'
  exit 1
fi

echo "Copying RST files for spec..."
cp -a $RST_SOURCE_DIR/spec/. $SPEC_SOURCE_DIR/spec

if [ -d "$SPEC_RST_HTML_BUILD_DIR" ]
then
  echo "Removing $SPEC_RST_HTML_BUILD_DIR"
  rm -r $SPEC_RST_HTML_BUILD_DIR
fi

echo "Generating HTML using Sphinx..."
sphinx-build -b html $SPEC_SOURCE_DIR $SPEC_RST_HTML_BUILD_DIR

echo "Removing files from build/html"

rm -r $SPEC_RST_HTML_BUILD_DIR/.doctrees
rm -r $SPEC_RST_HTML_BUILD_DIR/.buildinfo
