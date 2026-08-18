//===-- Utils.cpp -----------------------------------------------*- c++ -*-===//
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
// This file implements utilities for the KTDF dialect.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Support/LLVM.h>

#include <utility>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#define DEBUG_TYPE "ktdf-utilities"

using namespace mlir;
using namespace mlir::ktdf;

void mlir::ktdf::collectStages(Operation* root,
                               SmallVectorImpl<StageOp>& stages) {
  root->walk([&](StageOp stage) { stages.push_back(stage); });
}

void mlir::ktdf::collectQueriedUnits(Operation* root,
                                     SmallVectorImpl<Operation*>& query_ops) {
  root->walk([&](uniform::QueryMapOp op) { query_ops.push_back(op); });
}

auto mlir::ktdf::isTransferTarget(Value memref, Operation* root_op) -> bool {
  if (!root_op) {
    return false;
  }

  return root_op
      ->walk([&](DataTransferOp transfer) {
        // Check if memref is the destination
        if (transfer.getDestination() == memref) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

namespace {

auto isUsedInRegion(Value value, Region& region) -> bool {
  return llvm::any_of(value.getUsers(), [&](Operation* user) -> bool {
    return region.isAncestor(user->getParentRegion());
  });
}

}  // namespace

auto mlir::ktdf::findStageForUnit(uniform::QueryMapOp query_op,
                                  ArrayRef<StageOp> stages) -> StageOp {
  auto unit = query_op->getResult(0);

  for (auto stage : stages) {
    if (isUsedInRegion(unit, stage.getBodyRegion())) {
      return stage;
    }
  }

  return {};
}

auto mlir::ktdf::findLoadStage(ktdf::PipelineOp pipeline,
                               ktdf::StageOp compute_stage) -> StageOp {
  for (Value tok : compute_stage.getDependsIn()) {
    for (auto sibling : pipeline.getStages()) {
      if (sibling == compute_stage) continue;
      for (Value out_tok : sibling.getDependsOut())
        if (out_tok == tok) return sibling;
    }
  }
  return {};
}

auto mlir::ktdf::findStoreStage(ktdf::PipelineOp pipeline,
                                ktdf::StageOp compute_stage) -> StageOp {
  for (Value tok : compute_stage.getDependsOut()) {
    for (auto sibling : pipeline.getStages()) {
      if (sibling == compute_stage) continue;
      for (Value in_tok : sibling.getDependsIn())
        if (in_tok == tok) return sibling;
    }
  }
  return {};
}

auto mlir::ktdf::findLoadTransfer(ktdf::StageOp load_stage,
                                  linalg::GenericOp generic_op)
    -> std::pair<ktdf::DataTransferOp, ktdf::ReadFromFifoOp> {
  auto read_from_fifo =
      generic_op.getInputs().front().getDefiningOp<ktdf::ReadFromFifoOp>();
  if (!read_from_fifo) return {nullptr, nullptr};

  Value fifo_in = read_from_fifo.getFifoSlot();
  ktdf::DataTransferOp load_transfer;
  load_stage.getBody()->walk([&](ktdf::DataTransferOp dt) {
    if (dt.isSourceMemRef() && dt.isDestFifo() &&
        dt.getDestination() == fifo_in) {
      load_transfer = dt;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return {load_transfer, read_from_fifo};
}

auto mlir::ktdf::findStoreTransfer(ktdf::StageOp compute_stage,
                                   ktdf::StageOp store_stage,
                                   linalg::GenericOp generic_op)
    -> std::pair<ktdf::DataTransferOp, ktdf::WriteToFifoOp> {
  ktdf::WriteToFifoOp write_to_fifo;
  compute_stage.getBody()->walk([&](ktdf::WriteToFifoOp write) {
    if (write.getData() == generic_op.getResult(0)) {
      write_to_fifo = write;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!write_to_fifo) return {nullptr, nullptr};

  Value fifo_out = write_to_fifo.getFifoSlot();
  ktdf::DataTransferOp store_transfer;
  store_stage.getBody()->walk([&](ktdf::DataTransferOp dt) {
    if (dt.isSourceFifo() && dt.isDestMemRef() && dt.getSource() == fifo_out) {
      store_transfer = dt;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return {store_transfer, write_to_fifo};
}
