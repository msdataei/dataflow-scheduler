//===-- ReductionChunkAnalysis.h --------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// ReductionChunkAnalysis: given a linalg.generic with reduction iterators and
// a per-chunk byte budget, determine the minimum number of sequential chunks
// needed so that each chunk fits within the budget.
//
// The analysis inspects the static element type and shape of the first input
// tensor.  It finds the smallest integer N ≥ 1 such that:
//   1. (total_input_bytes / N) ≤ chunk_size_threshold
//   2. N evenly divides every reduction-dimension size
//
// If no such N exists (e.g. the tensor has dynamic dimensions, or the
// threshold is smaller than a single-element chunk) the result is
// std::nullopt.
//
// Example usage:
//   ReductionChunkResult result =
//       analyzeReductionChunks(generic_op, /*threshold_bytes=*/1 << 20);
//   if (result) {
//     unsigned n = result->num_chunks;
//     ArrayRef<int64_t> sizes = result->chunk_sizes;
//   }
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REDUCTIONCHUNKANALYSIS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REDUCTIONCHUNKANALYSIS_H_

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::ktdf {

/// Result returned by analyzeReductionChunks().
struct ReductionChunkResult {
  /// Number of sequential chunks (same value applied to every reduction dim).
  unsigned num_chunks;

  /// Per-reduction-dimension chunk size (dim_size / num_chunks), in the same
  /// order as the reduction iterators appear in the generic's iterator-type
  /// list (i.e. positions where the iterator type is `reduction`).
  llvm::SmallVector<int64_t> chunk_sizes;

  /// Indices (into the generic's iterator-type list) of the reduction dims.
  llvm::SmallVector<int64_t> reduction_dims;
};

/// Analyse a linalg.generic and return chunking parameters so that each chunk
/// fits within `chunk_size_threshold` bytes.
///
/// Returns std::nullopt when:
///   - the generic has no reduction iterators
///   - the first input tensor has a dynamic dimension
///   - the element type has no fixed bit-width (e.g. index type)
///   - no valid N ≤ product-of-reduction-dim-sizes satisfies both constraints
///
/// \param generic_op        The linalg.generic to analyse.
/// \param chunk_size_threshold  Per-chunk budget in bytes (default 1 MiB).
std::optional<ReductionChunkResult> analyzeReductionChunks(
    linalg::GenericOp generic_op, int64_t chunk_size_threshold = 1 << 20);

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REDUCTIONCHUNKANALYSIS_H_
