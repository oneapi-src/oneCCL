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

# Get the current working directory
SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# xml
XMLMainOneCCLvars="$SCRIPT_DIR/xml/group__OneCCLvars.xml"
XMLExpOneCCLvars="$SCRIPT_DIR/xml/group__ExpOneCCLvars.xml"
# man
MANMainOneCCLvars="$SCRIPT_DIR/man3/OneCCLvars.3"
MANExpOneCCLvars="$SCRIPT_DIR/man3/ExpOneCCLvars.3"
MANOneCCLvars="$SCRIPT_DIR/man3/OneCCL.3"
# MD
MDmain="$SCRIPT_DIR/main.md"
MDexp="$SCRIPT_DIR/exp.md"
MDdoc="$SCRIPT_DIR/OneCCL.md"

# Set default values for options
REMOVE_FILES=1

check_file_exists() {
  if [ ! -f "$1" ]; then
    echo "Error: $1 not found." >&2
    exit 1
  fi
}

check_program_exists() {
  if ! which "$1";  then
    echo "Error: $1 not found." >&2
    exit 1
  fi
}

# Define a function to print help message
print_help() {
  echo "Description: This script extracts information from two XML files, converts it to Markdown format, and combines it into"
  echo "             a single documentation file. It also combines two man files into one. By default, the intermediate Markdown"
  echo "             files and XML folder are deleted after processing, but you can use the -r option to keep them."
  echo ""
  echo "Usage: merge_docs.sh [-r 0|1] [-h]"
  echo "       -r: remove intermediate files (0: don't remove, 1: remove, default: 1)"
  echo "       -h: print this help message"
  echo ""
}

# Parse command-line options
while getopts ":r:h" opt; do
  case $opt in
    r)
      REMOVE_FILES=$OPTARG
      ;;
    h)
      print_help
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Define a function to extract relevant information from an XML file and save it in a Markdown file
extract_xml_info() {
  $SCRIPT_DIR/doxy_to_md.py $1 > $2
}

# generate docs
check_program_exists doxygen
doxygen "$SCRIPT_DIR/doxconfig"

# Combine two man files into one
check_file_exists "$MANMainOneCCLvars"
check_file_exists "$MANExpOneCCLvars"
cat "$MANMainOneCCLvars" "$MANExpOneCCLvars" > "$MANOneCCLvars"

# Extract information from two XML files and save them in two Markdown files
check_file_exists "$XMLMainOneCCLvars"
check_file_exists "$XMLExpOneCCLvars"
extract_xml_info "$XMLMainOneCCLvars" "$MDmain"
extract_xml_info "$XMLExpOneCCLvars" "$MDexp"

# Combine the two Markdown files into one
check_file_exists "$MDmain"
check_file_exists "$MDexp"
cat "$MDmain" "$MDexp" > "$MDdoc"

# Remove intermediate files if specified
if [ "$REMOVE_FILES" -eq 1 ]; then
  rm -f "$MANMainOneCCLvars" "$MANExpOneCCLvars"
  rm -f "$MDmain" "$MDexp"
  rm -rf "$SCRIPT_DIR/xml"
fi

# Remove type declarations and leave only names of environment variables
# Synopsis:
sed -Ei 's/constexpr const char \* \\fB([A-Z_]+)\\fP( = \x27[A-Z_]+\x27)?/\\fB\1\\fP/' $MANOneCCLvars
# Variable documentation
sed -Ei 's/constexpr const char\* ([A-Z_]+)( = \x27[A-Z_]+\x27)?\\fC \[constexpr\]\\fP/\1/' $MANOneCCLvars
