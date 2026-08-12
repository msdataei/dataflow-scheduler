//===------------------------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SIGNALINSERTION_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SIGNALINSERTION_H_

#include <map>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/StageToUnitsMap.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

namespace scheduler {

/// Metadata for a cross-iteration pipeline back-edge.
/// store_stage writes a shared scratchpad; load_stage reads it on the next
/// iteration.  The enclosing scf.for is stored so the signal guards can be
/// built from its lower-bound and upper-bound/step.
struct BackEdgeInfo {
  mlir::Operation* store_stage;  ///< leaf stage (producer, writes scratchpad)
  mlir::Operation* load_stage;   ///< root stage (consumer, reads scratchpad)
  mlir::scf::ForOp for_op;       ///< enclosing loop that carries the back-edge
};

/// Insert signal operations for all scratchpad conflicts found in global_dag.
///
/// For normal (intra-iteration) edges a single unconditional SignalOp is
/// placed after the producer stage.
///
/// For cross-iteration back-edges recorded in `back_edges` two guarded
/// SignalOps are emitted instead:
///   - at the start of load_stage body:  scf.if (iv != lb)  { signal }
///   - at the end   of store_stage body: scf.if (iv != ub-step) { signal }
mlir::LogicalResult insertSignals(
    mlir::Location loc, const StageToUnitsMap& stage_to_units,
    const mlir::ktdf::StageDependencyDAG& global_dag,
    const std::map<std::pair<mlir::Operation*, mlir::Operation*>,
                   llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts,
    const llvm::SmallVector<BackEdgeInfo>& back_edges,
    mlir::OpBuilder& builder);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SIGNALINSERTION_H_
