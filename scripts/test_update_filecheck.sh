#!/bin/bash

#
# Copyright 2026 The KTIR Scheduler Authors.
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

# Your mlir test file need to have at least one of the
# following RUN lines:
#
# RUN: scheduler --transform %s

# ---------------------------------------------------------------------------
# Derive the scheduler binary path and LLVM monorepo source root from a
# build directory.  Both are required to run the script.
#   CMakeCache.txt  →  LLVM_DIR  →  LLVMConfig.cmake  →  LLVM_BUILD_MAIN_SRC_DIR
#   LLVM_PROJ_SRC = parent of LLVM_BUILD_MAIN_SRC_DIR  (the llvm-project root)
# ---------------------------------------------------------------------------
function derivePathsFromBuildDir() {
  local build_dir="$1"
  local cmake_cache="${build_dir}/CMakeCache.txt"

  if [[ ! -f "${cmake_cache}" ]]; then
    echo "Error: CMakeCache.txt not found in '${build_dir}'"
    exit 1
  fi

  # Both driver binaries live in the same bin/ directory.
  # The actual binary to invoke is determined later per-file from the RUN line.
  SCHEDULER_BIN_DIR="${build_dir}/bin"
  # Keep a default for callers that don't go through run_binary detection.
  KTIRScheduler="${SCHEDULER_BIN_DIR}/dataflow-scheduler"

  # Extract the LLVM cmake directory recorded during the build.
  # CMakeCache.txt line format:  LLVM_DIR:PATH=<path>
  local llvm_cmake_dir
  llvm_cmake_dir=$(grep -m1 '^LLVM_DIR:PATH=' "${cmake_cache}" | cut -d= -f2-)
  if [[ -z "${llvm_cmake_dir}" ]]; then
    echo "Error: LLVM_DIR not found in '${cmake_cache}'"
    exit 1
  fi

  local llvm_config="${llvm_cmake_dir}/LLVMConfig.cmake"
  if [[ ! -f "${llvm_config}" ]]; then
    echo "Error: LLVMConfig.cmake not found at '${llvm_config}'"
    exit 1
  fi

  # LLVMConfig.cmake contains a line like:
  #   set(LLVM_BUILD_MAIN_SRC_DIR "/path/to/llvm-project/llvm")
  local llvm_src_dir
  llvm_src_dir=$(grep -m1 'LLVM_BUILD_MAIN_SRC_DIR' "${llvm_config}" \
                   | sed 's/.*"\(.*\)".*/\1/')
  if [[ -z "${llvm_src_dir}" ]]; then
    echo "Error: LLVM_BUILD_MAIN_SRC_DIR not found in '${llvm_config}'"
    exit 1
  fi

  # LLVM_BUILD_MAIN_SRC_DIR points to the llvm/ sub-directory of the
  # monorepo, so the monorepo root (which contains mlir/) is its parent.
  LLVM_PROJ_SRC="$(dirname "${llvm_src_dir}")"
}

# Usage
function printUsage() {
  echo "Usage: $(basename $0): "
  echo "          Run command:"
  echo "          ./scripts/test_update_filecheck.sh --build-dir=<path> --testfiles=<file_name.mlir>"
  echo
  echo "          --build-dir   path to the scheduler build directory (required). The"
  echo "                        dataflow-scheduler binary and LLVM source tree are derived"
  echo "                        automatically from it via CMakeCache.txt."
  echo "          --testfiles/-f name of the test file(s). This option can accept a directory too."
  echo
  echo "          Example:"
  echo "          ./scripts/test_update_filecheck.sh --build-dir=build -f=test/Dialects/basic.mlir"
  echo "          ./scripts/test_update_filecheck.sh --build-dir=build --testfiles=test/Dialects/basic.mlir"
  echo
  echo "          Note that your mlir test file need to have at least one of the following RUN lines:"
  echo "          // RUN: dataflow-scheduler <specific-pass-option> %s"
}

function parseCommandLine() {
  local build_dir=""

  for i in "$@"; do
    case $i in
      -f=*|--testfiles=*|--testfile=*)
        TESTFILES=${i#*=}
        ;;
      --build-dir=*)
        build_dir="${i#*=}"
        ;;
      -h|--help)
        printUsage;
        exit;
        ;;
      *)
        echo "Unrecognized option: $i"
        printUsage;
        exit 1;
        ;;
   esac
  done
  [[ -z "$@" ]] && printUsage && exit 1

  if [[ -z "${build_dir}" ]]; then
    echo "Error: --build-dir is required"
    printUsage
    exit 1
  fi

  derivePathsFromBuildDir "${build_dir}"
}

# ---------------------------------------------------------------------------
# process_run_line FILE RUN_LINE
#
# Runs the binary named in RUN_LINE against FILE, calls
# generate-test-checks.py with the appropriate --prefix, and prints the
# resulting CHECK block to stdout.  The original RUN line (with %s replaced
# by the actual path) is NOT emitted here; callers collect and prepend it.
# ---------------------------------------------------------------------------
function process_run_line() {
  local file="$1"
  local run_line="$2"

  # Strip leading // and RUN: marker.
  local stripped
  stripped=$(echo "${run_line}" | sed -e 's|^[[:space:]]*//__RUN__:||' \
                                      -e 's|^[[:space:]]*//[[:space:]]*RUN:[[:space:]]*||')

  # Detect round-trip (binary ... | binary ...).
  local is_roundtrip=0
  if echo "${stripped}" | grep -qE 'scheduler[^|]*\|[^|]*scheduler'; then
    is_roundtrip=1
  fi

  # Extract the binary name (first token).
  local run_binary
  run_binary=$(echo "${stripped}" | awk '{print $1}')

  # Resolve binary path.
  local resolved_binary="${SCHEDULER_BIN_DIR}/${run_binary}"
  if [[ ! -x "${resolved_binary}" ]]; then
    echo "Error: binary '${resolved_binary}' not found or not executable" >&2
    return 1
  fi

  # Extract flags: everything between the binary name and the first | (or end).
  local flags
  if [[ ${is_roundtrip} -eq 1 ]]; then
    flags=$(echo "${stripped}" | sed -e "s|^${run_binary}[[:space:]]*||" \
                                     -e 's/|.*//' \
                                     -e 's/%s//' \
                                     -e 's/[[:space:]]*$//')
  else
    flags=$(echo "${stripped}" | sed -e "s|^${run_binary}[[:space:]]*||" \
                                     -e 's/|[^|]*$//' \
                                     -e 's/FileCheck[^%]*%s[^%]*//' \
                                     -e 's/%s//' \
                                     -e 's/[[:space:]]*$//')
  fi
  flags=$(echo "${flags}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

  # Extract --check-prefix value (default: CHECK).
  local prefix
  prefix=$(echo "${stripped}" | grep -oE -- '--check-prefix=[^[:space:]]+' \
           | sed 's/--check-prefix=//' | head -1)
  [[ -z "${prefix}" ]] && prefix="CHECK"

  echo "  run command : ${flags}" >&2
  echo "  check prefix: ${prefix}" >&2

  # Run the pass and generate checks.
  local tmp_checks="/tmp/out_checks_${prefix}"
  if [[ -n "${flags}" ]]; then
    eval "${resolved_binary}" ${flags} "${file}" \
      | "${LLVM_PROJ_SRC}/mlir/utils/generate-test-checks.py" \
          --check-prefix="${prefix}" \
      > "${tmp_checks}"
  else
    eval "${resolved_binary}" "${file}" \
      | "${LLVM_PROJ_SRC}/mlir/utils/generate-test-checks.py" \
          --check-prefix="${prefix}" \
      > "${tmp_checks}"
  fi

  # generate-test-checks.py emits affine-map / affine-set attribute definition
  # lines with the bare "// CHECK:" prefix even when --check-prefix is set.
  # Replace them with the actual prefix so each run is self-contained.
  if [[ "${prefix}" != "CHECK" ]]; then
    sed -i '' -e "s|^// CHECK: #|// ${prefix}: #|g" "${tmp_checks}"
  fi

  # Convert CHECK:      (6 spaces) → CHECK-NEXT: for body lines.
  sed -i '' -e "s|// ${prefix}:      |// ${prefix}-NEXT:|g" "${tmp_checks}"

  # The output always contains a top-level wrapper module { ... } that has a
  # nested module { on the very next line.  CHECK-NEXT on that first nested
  # line fails because FileCheck sees an intervening line.  Convert all
  # {PREFIX}-NEXT: lines in that first CHECK-LABEL block back to plain
  # {PREFIX}: so FileCheck uses non-consecutive matching for them.
  awk -v p="${prefix}" '
    BEGIN { in_outer=0 }
    $0 ~ ("^// " p "-LABEL:   module \\{$") { in_outer=1; print; next }
    in_outer && $0 ~ ("^// " p "-LABEL:") { in_outer=0 }
    in_outer {
      line=$0
      gsub("^// " p "-NEXT:", "// " p ":", line)
      print line; next
    }
    { print }
  ' "${tmp_checks}" > /tmp/out && mv /tmp/out "${tmp_checks}"

  # When the output is a doubly-nested module (module { module { ... } }),
  # the second LABEL line would fail with -NEXT because FileCheck sees an
  # intervening blank/comment line.  Detect this structurally: two consecutive
  # {PREFIX}-LABEL:   module { lines appear before any other LABEL, which
  # means the inner module must be expressed as plain CHECK: lines and a
  # closing } must be appended to close the outer module.
  local has_double_module
  has_double_module=$(awk -v p="${prefix}" '
    $0 ~ ("^// " p "-LABEL:   module \\{$") { count++ }
    count == 2 { print "yes"; exit }
  ' "${tmp_checks}")
  if [[ "${has_double_module}" == "yes" ]]; then
    awk -v p="${prefix}" \
      '!done && $0 == "// " p "-LABEL:   module {" {
         print "// " p ": module {"
         print "// " p ":   module {"
         done=1; next
       } {print}' "${tmp_checks}" > /tmp/out && mv /tmp/out "${tmp_checks}"
    sed -i '' -e "s|^// ${prefix}-LABEL:   module {$|// ${prefix}:   module {|" \
        "${tmp_checks}"
    printf "// %s: }" "${prefix}" >> "${tmp_checks}"
  fi

  # Strip any preamble comment block generate-test-checks.py may have emitted.
  sed -e '/^\/\/ The script is designed to make adding checks to$/,/^\/\/ minimized and named to reflect the test intent\.$/d' \
      -e '/^\/\/ This script is intended to make adding checks to/,/^\/\/   \* https:\/\/mlir\.llvm\.org\/getting_started\/TestingGuide\/$/d' \
      "${tmp_checks}" > /tmp/out && mv /tmp/out "${tmp_checks}"
  # Also strip the NOTE header line generate-test-checks.py sometimes emits.
  sed -e "/\/\/ NOTE: Assertions have been autogenerated by/d" \
      "${tmp_checks}" > /tmp/out && mv /tmp/out "${tmp_checks}"

  cat "${tmp_checks}"
}

function run() {

  files="$(find $1 -name '*.mlir')"
  for file in $files ; do
    echo "test file: ${file}"

    # Collect all scheduler RUN lines from the file (bash 3.2 compatible).
    run_lines=()
    while IFS= read -r line; do
      run_lines+=("${line}")
    done < <(grep "RUN:" "${file}" | grep "scheduler")
    if [[ ${#run_lines[@]} -eq 0 ]]; then
      echo "  RUN line referencing a scheduler binary is needed — skipping"
      continue
    fi

    echo "  found ${#run_lines[@]} RUN line(s)"

    # Strip all existing RUN / CHECK / NOTE / preamble lines from the file so
    # we can rebuild them cleanly.
    sed -e "/^\/\/ RUN:/d" \
        -e "/\/\/ NOTE: Assertions have been autogenerated by/d" \
        -e "/\/\/ CHECK/d" \
        < "${file}" > /tmp/out && mv /tmp/out "${file}"
    # Remove any leftover preamble comment block from the MLIR body.
    sed -e '/^\/\/ The script is designed to make adding checks to$/,/^\/\/ minimized and named to reflect the test intent\.$/d' \
        -e '/^\/\/ This script is intended to make adding checks to/,/^\/\/   \* https:\/\/mlir\.llvm\.org\/getting_started\/TestingGuide\/$/d' \
        < "${file}" > /tmp/out && mv /tmp/out "${file}"

    # Generate check blocks for every RUN line and accumulate them.
    > /tmp/all_checks  # start fresh
    local all_run_headers=""
    for run_line in "${run_lines[@]}"; do
      # Reconstruct the canonical RUN header for this line.
      local stripped
      stripped=$(echo "${run_line}" | sed -e 's|^[[:space:]]*//[[:space:]]*RUN:[[:space:]]*||')
      local run_binary
      run_binary=$(echo "${stripped}" | awk '{print $1}')
      local is_roundtrip=0
      echo "${stripped}" | grep -qE 'scheduler[^|]*\|[^|]*scheduler' && is_roundtrip=1

      if [[ ${is_roundtrip} -eq 1 ]]; then
        local flags
        flags=$(echo "${stripped}" | sed -e "s|^${run_binary}[[:space:]]*||" \
                                         -e 's/|.*//' \
                                         -e 's/%s//' \
                                         -e 's/[[:space:]]*$//')
        flags=$(echo "${flags}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        all_run_headers+="// RUN: ${run_binary} ${flags} %s | ${run_binary} ${flags} | FileCheck %s"$'\n'
      else
        # Preserve the original RUN line as-is (it already has the right
        # --check-prefix and FileCheck invocation).
        all_run_headers+="// RUN: ${stripped}"$'\n'
      fi

      # Generate and append the check block.
      echo "--- processing: ${run_line}" >&2
      process_run_line "${file}" "${run_line}" >> /tmp/all_checks
      echo "" >> /tmp/all_checks  # blank separator between blocks
    done

    # The preamble comment to insert once, right after the RUN lines.
    local preamble
    preamble="// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/"

    # Prepend: RUN headers, preamble (once), then check blocks, then original MLIR body.
    {
      printf '%s\n' "${all_run_headers}"
      printf '%s\n' "${preamble}"
      echo ""
      cat /tmp/all_checks
    } | cat - "${file}" > /tmp/out && mv /tmp/out "${file}"

  done
}

# main
function main() {
  parseCommandLine "$@"
  run "$TESTFILES"
  return 0
}

main "$@"
