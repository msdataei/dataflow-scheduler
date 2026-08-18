//===-- Utils.h -------------------------------------------------*- c++ -*-===//
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
// This file declares utilities for the KTDF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_UTILS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_UTILS_H_

#include <llvm/ADT/SmallVector.h>

#include <utility>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::ktdf {

/// Collect all stages nested under a root via walk
void collectStages(Operation* root, SmallVectorImpl<StageOp>& stages);

/// Collect all queried units (uniform.query_map results)
void collectQueriedUnits(Operation* root,
                         SmallVectorImpl<Operation*>& query_ops);

/// Check if a memref value is written to within the given operation's region.
/// Returns true if any ktdf.data_transfer operation writes to the memref as a
/// destination.
/// \param memref The memref value to check
/// \param root_op The root operation to search within
/// \return true if the memref is written to, false otherwise
auto isTransferTarget(Value memref, Operation* root_op) -> bool;

/// Find stage for a given queried unit
// FIXME: This function has no users.
auto findStageForUnit(uniform::QueryMapOp query_op,
                      ArrayRef<ktdf::StageOp> stages) -> StageOp;

/// Find the sibling stage of `compute_stage` inside `pipeline` that is
/// upstream (load) of `compute_stage` — i.e. the sibling whose
/// `depends_out` token appears in `compute_stage`'s `depends_in`.
/// Returns a null StageOp if not found.
auto findLoadStage(ktdf::PipelineOp pipeline, ktdf::StageOp compute_stage)
    -> StageOp;

/// Find the sibling stage of `compute_stage` inside `pipeline` that is
/// downstream (store) of `compute_stage` — i.e. the sibling that
/// consumes a token from `compute_stage`'s `depends_out`.
/// Returns a null StageOp if not found.
auto findStoreStage(ktdf::PipelineOp pipeline, ktdf::StageOp compute_stage)
    -> StageOp;

/// Trace the input data path for a reduction linalg.generic:
///   generic.inputs[0] ← ReadFromFifoOp(fifo_in) ←
///   DataTransferOp(src=input_memref, dst=fifo_in) in load_stage
///
/// Returns {load_transfer, read_from_fifo}, or {nullptr, nullptr} on failure.
auto findLoadTransfer(ktdf::StageOp load_stage, linalg::GenericOp generic_op)
    -> std::pair<ktdf::DataTransferOp, ktdf::ReadFromFifoOp>;

/// Trace the output data path for a reduction linalg.generic:
///   generic.result(0) → WriteToFifoOp(fifo_out) →
///   DataTransferOp(src=fifo_out, dst=partial_memref) in store_stage
///
/// Returns {store_transfer, write_to_fifo}, or {nullptr, nullptr} on failure.
auto findStoreTransfer(ktdf::StageOp compute_stage, ktdf::StageOp store_stage,
                       linalg::GenericOp generic_op)
    -> std::pair<ktdf::DataTransferOp, ktdf::WriteToFifoOp>;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_UTILS_H_
