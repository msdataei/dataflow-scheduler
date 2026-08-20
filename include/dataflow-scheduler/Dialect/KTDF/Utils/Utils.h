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

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_UTILS_H_
