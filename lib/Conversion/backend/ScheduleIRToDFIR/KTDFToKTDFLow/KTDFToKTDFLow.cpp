//===----------------------------------------------------------------------===//
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
/// Phase 1: Operand-Based KTDF-to-DFIR Lowering
///
/// Converts KTDF pipelines/stages with applicable_units attributes to
/// operand-based ktdf_lowering IR using dataflow.get_unit, uniform maps,
/// and uniform queries.
///
//
//===----------------------------------------------------------------------===//

#include <map>

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ComponentClassifier.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/PipelineExecutionTransform.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ScratchpadConflicts.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/SignalInsertion.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UniformInfra.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UnitMaterializer.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "ktdf-to-ktdflowering"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTDFTOKTDFLOWERINGPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

namespace {

// ---------------------------------------------------------------------------
// Augment global_dag with cross-iteration back-edges and populate
// back_edges_out for every pipeline whose linalg.generic compute stage sits
// inside a non-trivial scf.for that is itself enclosed by a ktdf.parallel.
//
// Search rule (walking up from the pipeline):
//   - Stop at ktdf.parallel or func::FuncOp — the loop must be between the
//     pipeline and the ktdf.parallel, i.e. inside the parallel body.
//   - A loop found before hitting that boundary is the enclosing_for.
//   - Skip if the loop has a static trip count <= 1 (no back-edge needed).
//
// For each qualifying pipeline the back-edge is:
//   store_stage (token-successor of compute_stage) → load_stage (predecessor)
// injected into global_dag so computeScratchpadConflicts fires on the edge,
// while back_edges_out lets insertSignals emit guarded signals instead of the
// unconditional post-producer signal used for normal edges.
// ---------------------------------------------------------------------------
static void addScfForPipelineBackEdges(
    mlir::func::FuncOp func, mlir::ktdf::StageDependencyDAG& global_dag,
    llvm::SmallVector<BackEdgeInfo>& back_edges_out) {
  func.walk([&](mlir::linalg::GenericOp generic) {
    // Step 1: find the immediately enclosing ktdf.stage (compute_stage).
    mlir::Operation* p = generic->getParentOp();
    while (p && !mlir::isa<mlir::ktdf::StageOp>(p)) p = p->getParentOp();
    auto compute_stage = mlir::dyn_cast<mlir::ktdf::StageOp>(p);
    assert(compute_stage && "linalg.generic must be inside a ktdf.stage");

    // Step 2/3: walk up from compute_stage's parent pipeline looking for an
    // scf.for.  Stop at ktdf.parallel or func::FuncOp — the loop must be
    // between the pipeline and the enclosing parallel.
    auto pipeline = compute_stage->getParentOfType<mlir::ktdf::PipelineOp>();
    assert(pipeline && "compute_stage must be inside a ktdf.pipeline");

    mlir::scf::ForOp enclosing_for;
    for (mlir::Operation* anc = pipeline->getParentOp(); anc;
         anc = anc->getParentOp()) {
      if (mlir::isa<mlir::ktdf::ParallelOp, mlir::func::FuncOp>(anc)) break;
      if (auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(anc)) {
        enclosing_for = for_op;
        break;
      }
    }
    // No backedges if no enclosing loop
    if (!enclosing_for) return;

    // Step 4: skip trivial loops (trip count known to be <= 1).
    if (auto tc = scheduler::getStaticTripCount(enclosing_for); tc && *tc <= 1)
      return;

    // Steps 5+6: look up load_stage and store_stage directly from global_dag.
    // compute_stage is a leaf node in global_dag (no nested pipeline), so its
    // predecessor entry is the load leaf stage and its successor is the store
    // leaf stage.
    mlir::Operation* compute_op = compute_stage.getOperation();

    auto pred_it = global_dag.predecessors.find(compute_op);
    if (pred_it == global_dag.predecessors.end() || pred_it->second.empty())
      return;
    auto* load_stage_op = pred_it->second.front();

    auto succ_it = global_dag.successors.find(compute_op);
    if (succ_it == global_dag.successors.end() || succ_it->second.empty())
      return;
    auto* store_stage_op = succ_it->second.front();

    LDBG(1) << "  Adding scf.for pipeline back-edge: store_stage -> "
               "load_stage (cross-iteration scratchpad RAW)";
    global_dag.successors[store_stage_op].push_back(load_stage_op);
    global_dag.predecessors[load_stage_op].push_back(store_stage_op);

    back_edges_out.push_back(
        BackEdgeInfo{store_stage_op, load_stage_op, enclosing_for});
  });
}

struct KTDFToKTDFLoweringPass
    : public impl::KTDFToKTDFLoweringPassBase<KTDFToKTDFLoweringPass> {
  KTDFToKTDFLoweringPass()
      : scheduler_ctx_(SchedulerExtContext::dummyContext()) {}

  KTDFToKTDFLoweringPass(const SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void runOnOperation() override {
    LDBG(1) << "========= " PASS_NAME " =========";
    mlir::ModuleOp module_op = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module_op->emitError(
          "Unable to import the device specification. This could happen if the "
          "device spec file is empty or contains multiple devices");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<arch_view::ResourceKinds>(*device);

    llvm::SmallVector<mlir::func::FuncOp, 4> funcs;
    module_op.walk([&](mlir::func::FuncOp func) {
      funcs.push_back(func);
      return mlir::WalkResult::skip();
    });
    for (auto func : funcs) {
      LDBG(1) << "Running " << PASS_NAME << " on " << func.getName();

      // Pre-compute stages for reuse across multiple steps
      llvm::SmallVector<mlir::ktdf::StageOp, 8> stages;
      mlir::ktdf::collectStages(func, stages);

      if (stages.empty()) {
        LDBG(1) << "  No stages found - skipped";
        continue;
      }

      // Step 1: Classify components
      ComponentClassifier classifier(func);
      ComponentClassification components;
      if (mlir::failed(classifier.classify(stages, components))) {
        return signalPassFailure();
      }

      // Step 2: Extract grid size
      int grid_size = 0;
      if (mlir::failed(extractGridSize(func, grid_size))) {
        return signalPassFailure();
      }

      // Step 3: Materialize units
      UnitSSAMap unit_ssa_map;
      mlir::OpBuilder builder(&func.getBody().front(),
                              func.getBody().front().begin());

      UnitMaterializer materializer(func);
      if (mlir::failed(materializer.materialize(components, grid_size,
                                                unit_ssa_map, builder))) {
        return signalPassFailure();
      }

      // Step 4: Create uniform maps and queries
      QueriedUnitsMap queried_units;
      UniformMapsStorage uniform_maps;

      UniformInfra uniform_infra(func);
      if (mlir::failed(uniform_infra.createMapsAndQueries(
              components, grid_size, unit_ssa_map, queried_units, uniform_maps,
              builder))) {
        return signalPassFailure();
      }

      // Step 5: Wire queried units to stages
      mlir::OpBuilder phase2_builder(func.getContext());

      // Wire queried units from Step 4 to stages based on applicable_units
      StageToUnitsMap stage_to_units;
      int wired_queries = 0;
      for (auto stage : stages) {
        auto applicable_units = stage.getApplicableUnitsAttr();
        assert(applicable_units && "Stage should have applicable units");

        for (auto component : applicable_units.getValue()) {
          // Check if stage is in a parallel region
          mlir::Operation* parallel_parent =
              mlir::ktdf::findParallelParent(stage);

          if (parallel_parent) {
            // Parallel stage: look in queried_units.parallel for all corelets
            auto parallel_op =
                mlir::dyn_cast<mlir::ktdf::ParallelOp>(parallel_parent);
            int num_corelets = parallel_op.getNumInstances();
            for (int corelet = 0; corelet < num_corelets; ++corelet) {
              auto parallel_key = std::make_pair(
                  std::make_pair(parallel_parent, component), corelet);
              auto query_it = queried_units.parallel.find(parallel_key);
              if (query_it != queried_units.parallel.end()) {
                stage_to_units.mapping[stage.getOperation()].push_back(
                    query_it->second);
                wired_queries++;
              }
            }
          } else {
            // Non-parallel stage: first try queried_units.non_parallel
            auto query_it = queried_units.non_parallel.find(component);
            if (query_it != queried_units.non_parallel.end()) {
              stage_to_units.mapping[stage.getOperation()].push_back(
                  query_it->second);
              wired_queries++;
            } else {
              // If not found in non_parallel, this component might be in a
              // parallel region Find any parallel region that has this
              // component and use those units
              for (auto& [parallel_op, parallel_comps] :
                   components.parallel_components_map) {
                if (parallel_comps.contains(component)) {
                  auto parallel_parent_op =
                      mlir::dyn_cast<mlir::ktdf::ParallelOp>(parallel_op);
                  int num_corelets = parallel_parent_op.getNumInstances();
                  for (int corelet = 0; corelet < num_corelets; ++corelet) {
                    auto parallel_key = std::make_pair(
                        std::make_pair(parallel_op, component), corelet);
                    auto parallel_query_it =
                        queried_units.parallel.find(parallel_key);
                    if (parallel_query_it != queried_units.parallel.end()) {
                      stage_to_units.mapping[stage.getOperation()].push_back(
                          parallel_query_it->second);
                      wired_queries++;
                    }
                  }
                  break;  // Found the component in a parallel region, stop
                          // searching
                }
              }
            }
          }
        }
      }

      if (wired_queries == 0) {
        func.emitError("no queries wired to stages");
        return signalPassFailure();
      }

      // Step 6: Build the global flat stage DAG once, spanning all nesting
      // levels. Nodes are leaf StageOps only; used for conflict detection (Step
      // 7) and signal insertion (Step 8).
      mlir::ktdf::StageDependencyDAG global_dag;
      if (mlir::failed(mlir::ktdf::buildGlobalStageDAG(func, global_dag))) {
        func.emitError("failed to build global stage DAG");
        return signalPassFailure();
      }

      // Step 6b: Augment global_dag with cross-iteration back-edges for every
      // pipeline whose linalg.generic compute stage sits inside a non-trivial
      // scf.for enclosed by a ktdf.parallel.  Also collects BackEdgeInfo so
      // insertSignals can emit loop-IV-guarded signals for these edges.
      llvm::SmallVector<BackEdgeInfo> back_edges;
      addScfForPipelineBackEdges(func, global_dag, back_edges);

      // Step 7: Compute scratchpad conflicts across all pipelines using the
      // global leaf-stage DAG.
      std::map<std::pair<mlir::Operation*, mlir::Operation*>,
               llvm::SmallVector<scheduler::ResourceType, 2>>
          conflicts;
      if (mlir::failed(computeScratchpadConflicts(stage_to_units, global_dag,
                                                  resource_kinds, conflicts))) {
        func.emitError("failed to compute scratchpad conflicts");
        return signalPassFailure();
      }
      LDBG(1) << "Number of scratchpad conflicts found: " << conflicts.size();

      // Step 8: Insert signal operations for all conflicting global DAG edges,
      // before any pipeline transformation mutates the IR.  Back-edge signals
      // are wrapped in scf.if guards (iv != lb / iv != ub-step).
      if (mlir::failed(insertSignals(func.getLoc(), stage_to_units, global_dag,
                                     conflicts, back_edges, phase2_builder))) {
        return signalPassFailure();
      }

      // Steps 9-12: Process each pipeline independently (innermost
      // first).
      llvm::SmallVector<mlir::ktdf::PipelineOp, 8> pipelines;
      func.walk([&](mlir::ktdf::PipelineOp pipeline) {
        pipelines.push_back(pipeline);
      });

      if (!pipelines.empty()) {
        LDBG(1) << "  Processing " << pipelines.size() << " pipelines";

        // Process pipelines in reverse order (post-order walk for nested
        // pipelines)
        for (auto pipeline : llvm::reverse(pipelines)) {
          LDBG(1) << "  Processing pipeline at " << pipeline.getLoc() << "";

          // Collect stages that are direct children of this pipeline (not
          // nested)
          llvm::SmallVector<mlir::ktdf::StageOp, 8> pipeline_stages;
          for (auto& op : pipeline.getBodyRegion().front()) {
            if (auto stage = mlir::dyn_cast<mlir::ktdf::StageOp>(op)) {
              pipeline_stages.push_back(stage);
            }
          }

          if (pipeline_stages.empty()) {
            // Skip pipelines that have no direct stages (already transformed or
            // empty)
            LDBG(1) << "  Skipping pipeline with no direct stages";
            continue;
          }

          // Step 9: Analyze per-pipeline stage dependencies for topo-sort.
          mlir::ktdf::StageDependencyDAG dag;
          if (mlir::failed(
                  mlir::ktdf::analyzeStageDependencies(pipeline_stages, dag))) {
            return signalPassFailure();
          }

          // Step 10: Topologically sort stages
          llvm::SmallVector<mlir::ktdf::StageOp, 8> sorted_stages;
          if (mlir::failed(mlir::ktdf::topologicalSortStages(
                  pipeline_stages, dag, sorted_stages))) {
            pipeline.emitError("topological sort of stages failed");
            return signalPassFailure();
          }

          // Step 11: Transform stages to execute_on (inside-out: stages first)
          if (mlir::failed(transformStagesToExecuteOn(
                  pipeline, sorted_stages, stage_to_units, phase2_builder))) {
            return signalPassFailure();
          }

          // Step 12: Transform pipeline to execute_on (wrapping everything)
          if (mlir::failed(transformPipelineToExecuteOn(
                  pipeline, sorted_stages, stage_to_units, phase2_builder))) {
            return signalPassFailure();
          }

          // erasing the stages at the very end.
          for (auto& stage : sorted_stages) {
            stage.erase();
          }
        }

        LDBG(1) << "  Phase 2 complete";
      }

      // Remove loop_type attributes from all scf.for loops now that
      // lowering is complete and the attribute is no longer needed.
      func.walk(
          [](mlir::scf::ForOp for_op) { for_op->removeAttr("loop_type"); });

      LDBG(1) << "Lowering complete for " << func.getName() << "";
    }
  }

 private:
  const SchedulerExtContext& schedulerExtContext() const {
    return scheduler_ctx_;
  }

  const SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createKTDFToKTDFLoweringPass() {
  return std::make_unique<KTDFToKTDFLoweringPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createKTDFToKTDFLoweringPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<KTDFToKTDFLoweringPass>(scheduler_ctx);
}
