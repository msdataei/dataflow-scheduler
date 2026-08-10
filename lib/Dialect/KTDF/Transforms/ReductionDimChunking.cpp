//===-- ReductionDimChunking.cpp --------------------------------*- c++ -*-===//
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
// ReductionDimChunking: split the reduction dimension of the inner
// linalg.generic into sequential chunks so that each chunk fits in the
// hardware FIFO path.
//
// The pass operates on the shape produced by StageCoarsening.  It expects a
// top-level ktdf.pipeline with three sibling stages:
//
//   stage-0 (MNILU)  : DDR → L1 load
//   stage-1 (compute): outer scf.for (batch) whose body is a single inner
//                       ktdf.pipeline with three stages:
//                         stage-a (L1LU) : L1 → FIFO
//                         stage-b (SFU)  : linalg.generic reduce
//                         stage-c (L1SU) : FIFO → L1
//   stage-2 (MNISU)  : L1 → DDR store
//
// The transformation replaces the single inner ktdf.pipeline with a
// scf.for over all num_chunks iterations.  Each iteration contains one
// ktdf.pipeline (Load / SFU / Store stages).  First-vs-rest accumulation
// behaviour is selected at runtime via %is_first = (chunk_iv == 0):
//
//   Load stage : always transfers the input chunk slice to fifo_in.
//                When !is_first, additionally transfers the partial
//                accumulator (the existing L1 output buffer) to fifo_out
//                so the SFU can read it back.
//   SFU stage  : when is_first, initialises the output tensor with
//                tensor.empty; otherwise reads the partial result from
//                fifo_out.  The linalg.generic and write_to_fifo are
//                unconditional.
//   Store stage: unconditionally writes fifo_out back to the L1 output
//                buffer; no is_last distinction is needed.
//
// The existing L1 output buffer (discovered via the store stage's
// data_transfer destination) is reused as the partial-accumulation buffer
// — no new memref is allocated.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/ReductionChunkAnalysis.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "reduction-dim-chunking"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;
using namespace mlir::ktdf;

namespace mlir::ktdf {
#define GEN_PASS_DEF_REDUCTIONDIMCHUNKINGPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Reduction Dim Chunking pass"),
    llvm::cl::init(false));

namespace {

// ---------------------------------------------------------------------------
// Helper: find the unique linalg.generic with a reduction iterator inside a
// ktdf.stage body.  Returns nullptr if not found.
// ---------------------------------------------------------------------------
static linalg::GenericOp findReductionGenericOp(ktdf::StageOp stage) {
  linalg::GenericOp found;
  stage.getBody()->walk([&](linalg::GenericOp generic) {
    for (auto it : generic.getIteratorTypesArray()) {
      if (it == utils::IteratorType::reduction) {
        found = generic;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

// ---------------------------------------------------------------------------
// Helper: find the upstream (load) stage of a compute stage — the sibling
// whose depends_out token appears in compute_stage's depends_in.
// ---------------------------------------------------------------------------
static ktdf::StageOp findLoadStage(ktdf::PipelineOp pipeline,
                                   ktdf::StageOp compute_stage) {
  for (Value tok : compute_stage.getDependsIn()) {
    for (auto sibling : pipeline.getStages()) {
      if (sibling == compute_stage) continue;
      for (Value out_tok : sibling.getDependsOut())
        if (out_tok == tok) return sibling;
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Helper: find the downstream (store) stage of a compute stage — the sibling
// whose depends_in token appears in compute_stage's depends_out.
// ---------------------------------------------------------------------------
static ktdf::StageOp findStoreStage(ktdf::PipelineOp pipeline,
                                    ktdf::StageOp compute_stage) {
  for (Value tok : compute_stage.getDependsOut()) {
    for (auto sibling : pipeline.getStages()) {
      if (sibling == compute_stage) continue;
      for (Value in_tok : sibling.getDependsIn())
        if (in_tok == tok) return sibling;
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Helper: build a ktdf.private with FIFO slots + two tokens and yield them.
//
// Without partial FIFO slot (partial_slot_type == nullopt):
//   results: [fifo_in(#0), fifo_out(#1), tok0(#2), tok1(#3)]
//
// With an optional partial FIFO slot:
//   results: [fifo_in(#0), fifo_partial(#1), fifo_out(#2), tok0(#3), tok1(#4)]
//
// The single-pipeline design only ever passes nullopt here; the partial-
// slot overload is retained for potential future use.
// ---------------------------------------------------------------------------
static ktdf::PrivateOp buildFifoPrivate(
    OpBuilder& builder, Location loc, ktdf::FifoSlotType in_slot_type,
    ktdf::FifoSlotType out_slot_type,
    std::optional<ktdf::FifoSlotType> partial_slot_type) {
  auto tok_type = builder.getType<ktdf::TokenType>();
  SmallVector<Type> result_types;
  result_types.push_back(in_slot_type);
  if (partial_slot_type) result_types.push_back(*partial_slot_type);
  result_types.push_back(out_slot_type);
  result_types.push_back(tok_type);
  result_types.push_back(tok_type);

  auto priv_op = ktdf::PrivateOp::create(builder, loc, result_types);
  OpBuilder body_bldr(priv_op.getRegion());
  Location body_loc = loc;
  auto slot_in = ktdf::FifoAllocateOp::create(
      body_bldr, body_loc, TypeRange{in_slot_type}, ValueRange{});
  Value slot_partial_val;
  if (partial_slot_type) {
    auto slot_partial = ktdf::FifoAllocateOp::create(
        body_bldr, body_loc, TypeRange{*partial_slot_type}, ValueRange{});
    slot_partial_val = slot_partial.getResult(0);
  }
  auto slot_out = ktdf::FifoAllocateOp::create(
      body_bldr, body_loc, TypeRange{out_slot_type}, ValueRange{});
  auto tok0 = ktdf::CreateTokenOp::create(body_bldr, body_loc, tok_type);
  auto tok1 = ktdf::CreateTokenOp::create(body_bldr, body_loc, tok_type);

  SmallVector<Value> yield_vals;
  yield_vals.push_back(slot_in.getResult(0));
  if (slot_partial_val) yield_vals.push_back(slot_partial_val);
  yield_vals.push_back(slot_out.getResult(0));
  yield_vals.push_back(tok0.getResult());
  yield_vals.push_back(tok1.getResult());
  ktdf::PrivateYieldOp::create(body_bldr, body_loc, yield_vals);
  return priv_op;
}

// ---------------------------------------------------------------------------
// Main pass struct
// ---------------------------------------------------------------------------
struct ReductionDimChunkingPass
    : public ktdf::impl::ReductionDimChunkingPassBase<
          ReductionDimChunkingPass> {
  using ReductionDimChunkingPassBase<
      ReductionDimChunkingPass>::ReductionDimChunkingPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    ModuleOp module = getOperation();
    if (failed(transformModule(module))) signalPassFailure();
  }

 private:
  // -----------------------------------------------------------------------
  // Top-level transformation entry point.
  // Walk for linalg.generic ops with a reduction iterator; for each one,
  // find the immediately enclosing ktdf.stage (the SFU stage) and collect
  // unique stages before transforming to avoid mutating the IR while walking.
  // -----------------------------------------------------------------------
  LogicalResult transformModule(ModuleOp module) {
    SmallVector<ktdf::StageOp> compute_stages;
    module.walk([&](linalg::GenericOp generic) {
      // Check if this generic has a reduction iterator.
      bool has_reduction = false;
      for (auto it : generic.getIteratorTypesArray())
        if (it == utils::IteratorType::reduction) {
          has_reduction = true;
          break;
        }
      if (!has_reduction) return WalkResult::advance();

      // Walk up to find the immediately enclosing ktdf.stage.
      Operation* parent = generic->getParentOp();
      while (parent && !isa<ktdf::StageOp>(parent))
        parent = parent->getParentOp();
      if (!parent) return WalkResult::advance();

      auto stage = cast<ktdf::StageOp>(parent);
      // Avoid duplicates (multiple generics in same stage).
      if (llvm::find(compute_stages, stage) == compute_stages.end())
        compute_stages.push_back(stage);
      return WalkResult::advance();
    });

    for (auto compute_stage : compute_stages)
      if (failed(transformPipeline(compute_stage))) return failure();

    return success();
  }

  // -----------------------------------------------------------------------
  // Transform the outer pipeline given the SFU stage (the ktdf.stage that
  // directly contains the reduction linalg.generic).
  // -----------------------------------------------------------------------
  LogicalResult transformPipeline(ktdf::StageOp compute_stage) {
    // Navigate up: SFU stage → inner pipeline.
    auto inner_pipeline =
        dyn_cast<ktdf::PipelineOp>(compute_stage->getParentOp());
    if (!inner_pipeline) {
      LDBG(1) << PASS_NAME
          ": SFU stage not directly inside a pipeline — "
          "skipping";
      return success();
    }

    // Verify there is exactly one inner pipeline inside the parent region.
    unsigned inner_pipeline_count = 0;
    inner_pipeline->getParentRegion()->walk<WalkOrder::PreOrder>(
        [&](ktdf::PipelineOp) {
          ++inner_pipeline_count;
          return WalkResult::skip();
        });
    if (inner_pipeline_count != 1) {
      LDBG(1) << PASS_NAME ": expected exactly 1 inner pipeline, got "
              << inner_pipeline_count << " — skipping";
      return success();
    }

    ktdf::StageOp load_stage = findLoadStage(inner_pipeline, compute_stage);
    if (!load_stage) {
      LDBG(1) << PASS_NAME
          ": could not find load stage upstream of compute "
          "stage — skipping";
      return success();
    }

    ktdf::StageOp store_stage = findStoreStage(inner_pipeline, compute_stage);
    if (!store_stage) {
      LDBG(1) << PASS_NAME
          ": could not find store stage downstream of compute "
          "stage — skipping";
      return success();
    }

    // Locate the linalg.generic with a reduction iterator.
    linalg::GenericOp generic_op = findReductionGenericOp(compute_stage);
    if (!generic_op) {
      LDBG(1) << PASS_NAME
          ": no reduction linalg.generic in SFU stage — "
          "skipping";
      return success();
    }

    // ------------------------------------------------------------------
    // Determine reduction dims, num_chunks, and per-dim chunk sizes.
    //
    // When --num-chunks was provided by the user, validate and use it.
    // Otherwise delegate to ReductionChunkAnalysis which picks the
    // smallest N that fits within chunkSizeThreshold bytes per chunk.
    // ------------------------------------------------------------------
    SmallVector<int64_t> reduction_dims;
    SmallVector<int64_t> chunk_sizes;
    unsigned loop_num_chunks = 0;

    if (numChunks.empty()) {
      // Auto-infer via analysis.
      auto result = analyzeReductionChunks(generic_op, chunkSizeThreshold);
      if (!result) {
        LDBG(1) << PASS_NAME
                << ": ReductionChunkAnalysis could not infer chunk count — "
                   "skipping";
        return success();
      }
      loop_num_chunks = result->num_chunks;
      chunk_sizes = std::move(result->chunk_sizes);
      reduction_dims = std::move(result->reduction_dims);
    } else {
      // User-supplied --num-chunks path: collect reduction dims and validate.
      auto iter_types = generic_op.getIteratorTypesArray();
      for (int64_t i = 0; i < static_cast<int64_t>(iter_types.size()); ++i)
        if (iter_types[i] == utils::IteratorType::reduction)
          reduction_dims.push_back(i);

      if (reduction_dims.empty()) {
        LDBG(1) << PASS_NAME ": could not find reduction dimension — skipping";
        return success();
      }

      if (numChunks.size() != reduction_dims.size()) {
        LDBG(1) << PASS_NAME ": numChunks has " << numChunks.size()
                << " entries but there are " << reduction_dims.size()
                << " reduction dims — skipping";
        return success();
      }

      // Compute total (flat) chunk count = product of per-dim chunk counts.
      // Per-dim counts may differ; the single sequential loop iterates over
      // all combinations via stride-based de-interleaving.
      loop_num_chunks = 1;
      for (unsigned nc : numChunks) loop_num_chunks *= nc;

      auto input_type =
          cast<RankedTensorType>(generic_op.getInputs().front().getType());
      for (size_t j = 0; j < reduction_dims.size(); ++j) {
        int64_t dim = reduction_dims[j];
        int64_t dim_size = input_type.getDimSize(dim);
        if (dim_size == ShapedType::kDynamic) {
          LDBG(1) << PASS_NAME
                  << ": dynamic reduction size not yet supported — skipping";
          return success();
        }
        unsigned nchunk = numChunks[j];
        if (nchunk == 0 || dim_size % nchunk != 0) {
          LDBG(1) << PASS_NAME ": num_chunks=" << nchunk
                  << " does not evenly divide dim " << dim
                  << " size=" << dim_size << " — skipping";
          return success();
        }
        chunk_sizes.push_back(dim_size / static_cast<int64_t>(nchunk));
      }
    }

    if (reduction_dims.empty()) {
      LDBG(1) << PASS_NAME ": could not find reduction dimension — skipping";
      return success();
    }

    LDBG(1) << PASS_NAME ": num_reduction_dims=" << reduction_dims.size()
            << " num_chunks=" << loop_num_chunks;

    // Collect per-dim chunk counts in reduction-dim order.
    SmallVector<int64_t> per_dim_num_chunks;
    if (numChunks.empty()) {
      // Auto-inferred path: all dims use the same num_chunks.
      per_dim_num_chunks.assign(reduction_dims.size(),
                                static_cast<int64_t>(loop_num_chunks));
    } else {
      for (unsigned nc : numChunks)
        per_dim_num_chunks.push_back(static_cast<int64_t>(nc));
    }

    return rewriteComputeStage(inner_pipeline, load_stage, compute_stage,
                               store_stage, generic_op, reduction_dims,
                               chunk_sizes, per_dim_num_chunks,
                               static_cast<int64_t>(loop_num_chunks));
  }

  // -----------------------------------------------------------------------
  // Core rewrite: replace the inner pipeline with a single-loop chunked form.
  //
  // Instead of generating separate Phase-A / Phase-B-loop / Phase-C pipelines,
  // we emit a single scf.for over all chunks containing one ktdf.pipeline.
  // First-vs-rest behaviour is guarded at runtime with scf.if (%is_first).
  // -----------------------------------------------------------------------
  LogicalResult rewriteComputeStage(
      ktdf::PipelineOp inner_pipeline, ktdf::StageOp load_stage,
      ktdf::StageOp compute_stage, ktdf::StageOp store_stage,
      linalg::GenericOp generic_op, ArrayRef<int64_t> reduction_dims,
      ArrayRef<int64_t> chunk_sizes, ArrayRef<int64_t> per_dim_num_chunks,
      int64_t num_chunks) {
    MLIRContext* context = inner_pipeline.getContext();
    IRRewriter rewriter(context);
    Location loc = inner_pipeline.getLoc();

    // ------------------------------------------------------------------
    // Step 1: Reuse the existing outer-pipeline L1 output buffer for partial
    // accumulation.
    // ------------------------------------------------------------------
    ktdf::DataTransferOp store_transfer;
    store_stage.getBody()->walk([&](ktdf::DataTransferOp dt) {
      if (dt.isDestMemRef()) store_transfer = dt;
    });
    if (!store_transfer) {
      inner_pipeline.emitError(PASS_NAME
                               ": no memref-dest data_transfer in "
                               "store stage");
      return failure();
    }
    Value partial_memref = store_transfer.getDestination();
    auto output_memref_type = cast<MemRefType>(partial_memref.getType());

    // ------------------------------------------------------------------
    // Step 2: Collect constants we need for arithmetic in the new bodies.
    // Insert them before the batch for-loop so they are in scope everywhere.
    // ------------------------------------------------------------------
    scf::ForOp batch_for =
        dyn_cast<scf::ForOp>(inner_pipeline->getParentRegion()->getParentOp());
    rewriter.setInsertionPoint(batch_for ? batch_for.getOperation()
                                         : inner_pipeline.getOperation());
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Per-reduction-dim: chunk_size constants and num_chunks constants.
    SmallVector<Value> c_chunk_sizes;
    for (int64_t cs : chunk_sizes)
      c_chunk_sizes.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, cs));

    // Row-major strides for de-interleaving flat chunk_idx.
    SmallVector<int64_t> dim_strides(per_dim_num_chunks.size(), 1);
    for (int64_t j = static_cast<int64_t>(per_dim_num_chunks.size()) - 2;
         j >= 0; --j)
      dim_strides[j] = dim_strides[j + 1] * per_dim_num_chunks[j + 1];
    SmallVector<Value> c_dim_strides;
    SmallVector<Value> c_per_dim_num_chunks;
    for (size_t j = 0; j < per_dim_num_chunks.size(); ++j) {
      c_dim_strides.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, dim_strides[j]));
      c_per_dim_num_chunks.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, per_dim_num_chunks[j]));
    }
    Value c_num_chunks =
        arith::ConstantIndexOp::create(rewriter, loc, num_chunks);
    // c_num_chunks_m1 is not used directly in the new single-loop design;
    // we still emit it (and suppress the warning below) so the outer scope
    // retains the constant for any future reference.
    Value c_num_chunks_m1 =
        arith::ConstantIndexOp::create(rewriter, loc, num_chunks - 1);

    // ------------------------------------------------------------------
    // Step 3: Extract the input memref and derive the compute stage FIFO slot
    // types from the linalg.generic dataflow.
    // ------------------------------------------------------------------
    ktdf::DataTransferOp load_transfer;
    load_stage.getBody()->walk([&](ktdf::DataTransferOp transfer) {
      if (transfer.isSourceMemRef() && transfer.isDestFifo()) {
        load_transfer = transfer;
      }
    });
    if (!load_transfer) {
      inner_pipeline.emitError(PASS_NAME
                               ": no memref-to-fifo transfer in "
                               "load stage");
      return failure();
    }
    Value input_memref = load_transfer.getSource();

    auto read_from_fifo =
        generic_op.getInputs().front().getDefiningOp<ktdf::ReadFromFifoOp>();
    if (!read_from_fifo) {
      inner_pipeline.emitError(
          PASS_NAME ": generic input is not produced by read_from_fifo");
      return failure();
    }

    ktdf::WriteToFifoOp write_to_fifo;
    compute_stage.getBody()->walk([&](ktdf::WriteToFifoOp write) {
      if (write.getData() == generic_op.getResult(0)) {
        write_to_fifo = write;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!write_to_fifo) {
      inner_pipeline.emitError(
          PASS_NAME ": generic result is not consumed by write_to_fifo");
      return failure();
    }

    auto in_fifo_type =
        dyn_cast<ktdf::FifoSlotType>(read_from_fifo.getFifoSlot().getType());
    auto out_fifo_type =
        dyn_cast<ktdf::FifoSlotType>(write_to_fifo.getFifoSlot().getType());
    if (!in_fifo_type || !out_fifo_type) {
      inner_pipeline.emitError(
          PASS_NAME
          ": could not derive input/output FIFO slot types "
          "from compute stage");
      return failure();
    }

    // Per-chunk input tensor type: reduction dims replaced with chunk_sizes.
    auto orig_input_tensor_type =
        cast<RankedTensorType>(generic_op.getInputs().front().getType());
    SmallVector<int64_t> chunk_input_shape(
        orig_input_tensor_type.getShape().begin(),
        orig_input_tensor_type.getShape().end());
    for (size_t j = 0; j < reduction_dims.size(); ++j)
      chunk_input_shape[reduction_dims[j]] = chunk_sizes[j];
    auto chunk_input_tensor_type = RankedTensorType::get(
        chunk_input_shape, orig_input_tensor_type.getElementType());

    // Output tensor type (unchanged).
    auto output_tensor_type =
        cast<RankedTensorType>(generic_op.getOutputs().front().getType());

    // ------------------------------------------------------------------
    // Step 4: Build the replacement: a single scf.for over all chunks that
    // contains one ktdf.pipeline.  First-vs-rest behaviour is selected at
    // runtime with scf.if guards.
    //
    // We insert the chunk loop BEFORE inner_pipeline, then erase it.
    // ------------------------------------------------------------------
    rewriter.setInsertionPoint(inner_pipeline);

    auto input_memref_type = cast<MemRefType>(input_memref.getType());
    int64_t chunk_fifo_elements = chunk_input_tensor_type.getNumElements();
    auto chunk_in_fifo_type = ktdf::FifoSlotType::get(
        context, in_fifo_type.getSrc(), in_fifo_type.getDest(),
        chunk_fifo_elements, in_fifo_type.getElementType());

    // Applicable-units attributes from original stages.
    auto load_units = load_stage.getApplicableUnitsAttr();
    auto sfu_units = compute_stage.getApplicableUnitsAttr();
    auto store_units = store_stage.getApplicableUnitsAttr();

    // Batch-loop induction variable.
    Value batch_iv = batch_for.getInductionVar();

    // Emit L1-slot index: divsi(subi(batch_iv, c0), c1).
    auto emit_l1_idx = [&](OpBuilder& builder, Location l,
                           Value batch_iv_val) -> Value {
      Value sub = arith::SubIOp::create(builder, l, batch_iv_val, c0);
      return arith::DivSIOp::create(builder, l, sub, c1);
    };

    // ----------------------------------------------------------------
    // Emit constants that belong INSIDE the compute stage's body
    // (they reference chunk-related values and are best placed there).
    // We emit them just before the chunk for-loop so the loop body can
    // capture them as dominating definitions.
    // ----------------------------------------------------------------
    // These inner constants will be placed at the insertion point that is
    // now set to just before inner_pipeline (i.e., inside the batch for-loop
    // body, before the old inner pipeline).
    Value inner_c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value inner_c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value c_num_chunks_const =
        arith::ConstantIndexOp::create(rewriter, loc, num_chunks);
    // Per-dim chunk_size and num_chunks constants placed at the inner
    // insertion point (just before the chunk for-loop) so they dominate
    // the loop body.
    SmallVector<Value> inner_c_chunk_sizes;
    for (int64_t cs : chunk_sizes)
      inner_c_chunk_sizes.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, cs));
    SmallVector<Value> inner_c_dim_strides;
    SmallVector<Value> inner_c_per_dim_num_chunks;
    for (size_t j = 0; j < per_dim_num_chunks.size(); ++j) {
      inner_c_dim_strides.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, dim_strides[j]));
      inner_c_per_dim_num_chunks.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, per_dim_num_chunks[j]));
    }
    (void)arith::ConstantIndexOp::create(rewriter, loc, num_chunks - 1);

    // ---- Single chunk for-loop ----
    auto chunk_for = scf::ForOp::create(rewriter, loc, inner_c0,
                                        c_num_chunks_const, inner_c1);
    Value chunk_iv = chunk_for.getInductionVar();

    OpBuilder chunk_bldr(chunk_for.getBody(), chunk_for.getBody()->begin());

    // %is_first = arith.cmpi eq, %chunk, %c0
    Value is_first = arith::CmpIOp::create(
        chunk_bldr, loc, arith::CmpIPredicate::eq, chunk_iv, inner_c0);

    // ---- Single pipeline per chunk ----
    auto phase_pipeline = ktdf::PipelineOp::create(chunk_bldr, loc);
    OpBuilder pipe_bldr(phase_pipeline.getBody(),
                        phase_pipeline.getBody()->end());

    // Private: fifo_in, fifo_out, tok0, tok1 (same 4-result shape for all
    // chunks).
    auto slots = buildFifoPrivate(pipe_bldr, loc, chunk_in_fifo_type,
                                  out_fifo_type, std::nullopt);
    Value fifo_in = slots.getResult(0);
    Value fifo_out = slots.getResult(1);
    Value tok0 = slots.getResult(2);
    Value tok1 = slots.getResult(3);

    // -------- Load stage --------
    auto load_new = ktdf::StageOp::create(pipe_bldr, loc,
                                          /*depends_in=*/ValueRange{},
                                          /*depends_out=*/ValueRange{tok0});
    if (load_units) load_new.setApplicableUnitsAttr(load_units);
    {
      OpBuilder stage_bldr(load_new.getBody(), load_new.getBody()->end());
      Value l1_idx = emit_l1_idx(stage_bldr, loc, batch_iv);

      // Per-reduction-dim row offsets: (chunk_iv / stride[j]) % num_chunks[j]
      // * chunk_size[j].
      SmallVector<Value> row_offsets;
      for (size_t j = 0; j < inner_c_chunk_sizes.size(); ++j) {
        Value divided = arith::DivSIOp::create(stage_bldr, loc, chunk_iv,
                                               inner_c_dim_strides[j]);
        Value per_dim_idx = arith::RemSIOp::create(
            stage_bldr, loc, divided, inner_c_per_dim_num_chunks[j]);
        row_offsets.push_back(arith::MulIOp::create(
            stage_bldr, loc, per_dim_idx, inner_c_chunk_sizes[j]));
      }

      // Source indices for the input data transfer.
      int64_t input_rank = input_memref_type.getRank();
      SmallVector<Value> src_indices;
      src_indices.reserve(static_cast<size_t>(input_rank));
      src_indices.push_back(l1_idx);
      for (int64_t i = 1; i < input_rank; ++i) {
        int64_t generic_dim = i - 1;
        auto it = llvm::find(reduction_dims, generic_dim);
        if (it != reduction_dims.end())
          src_indices.push_back(row_offsets[it - reduction_dims.begin()]);
        else
          src_indices.push_back(inner_c0);
      }

      SmallVector<int64_t> src_static_sizes;
      src_static_sizes.push_back(1);
      for (int64_t i = 1; i < input_rank; ++i) {
        int64_t generic_dim = i - 1;
        auto it = llvm::find(reduction_dims, generic_dim);
        src_static_sizes.push_back(
            it != reduction_dims.end()
                ? chunk_sizes[it - reduction_dims.begin()]
                : input_memref_type.getDimSize(i));
      }

      AffineMap input_id = AffineMap::getMultiDimIdentityMap(
          input_rank, stage_bldr.getContext());
      ktdf::DataTransferOp::create(stage_bldr, loc, input_memref, input_id,
                                   src_indices, src_static_sizes, fifo_in,
                                   AffineMap{}, ValueRange{},
                                   ArrayRef<int64_t>{chunk_fifo_elements});

      // scf.if %is_first {} else { load partial buffer into fifo_out }
      auto if_op =
          scf::IfOp::create(stage_bldr, loc, /*resultTypes=*/TypeRange{},
                            is_first, /*withElseRegion=*/true);
      // then-region: empty — auto-yield already inserted by builder.
      // else-region: transfer partial buffer → fifo_out (before auto-yield).
      {
        OpBuilder else_bldr =
            OpBuilder::atBlockBegin(&if_op.getElseRegion().front());
        int64_t partial_rank = output_memref_type.getRank();
        AffineMap partial_id = AffineMap::getMultiDimIdentityMap(
            partial_rank, else_bldr.getContext());
        SmallVector<Value> partial_src_indices;
        partial_src_indices.push_back(l1_idx);
        for (int64_t i = 1; i < partial_rank; ++i) {
          partial_src_indices.push_back(inner_c0);
        }
        SmallVector<int64_t> partial_src_sizes;
        partial_src_sizes.push_back(1);
        for (int64_t i = 1; i < partial_rank; ++i) {
          partial_src_sizes.push_back(output_memref_type.getDimSize(i));
        }
        ktdf::DataTransferOp::create(
            else_bldr, loc, partial_memref, partial_id, partial_src_indices,
            partial_src_sizes, fifo_out, AffineMap{}, ValueRange{},
            ArrayRef<int64_t>{
                static_cast<int64_t>(out_fifo_type.getNumElements())});
      }
    }

    // -------- SFU stage --------
    auto sfu_new = ktdf::StageOp::create(pipe_bldr, loc,
                                         /*depends_in=*/ValueRange{tok0},
                                         /*depends_out=*/ValueRange{tok1});
    if (sfu_units) sfu_new.setApplicableUnitsAttr(sfu_units);
    {
      OpBuilder stage_bldr(sfu_new.getBody(), sfu_new.getBody()->end());

      // Read input chunk from fifo_in.
      auto in_tensor = ktdf::ReadFromFifoOp::create(
          stage_bldr, loc, chunk_input_tensor_type, fifo_in);

      // scf.if %is_first -> tensor<1x64xf16> {
      //   tensor.empty()
      // } else {
      //   read_from_fifo fifo_out
      // }
      auto if_outs = scf::IfOp::create(stage_bldr, loc,
                                       TypeRange{output_tensor_type}, is_first,
                                       /*withElseRegion=*/true);
      {
        // then: zero-init
        OpBuilder then_bldr =
            OpBuilder::atBlockBegin(&if_outs.getThenRegion().front());
        Value empty_tensor = tensor::EmptyOp::create(
            then_bldr, loc, output_tensor_type.getShape(),
            output_tensor_type.getElementType());
        scf::YieldOp::create(then_bldr, loc, empty_tensor);
      }
      {
        // else: read accumulated partial from fifo_out
        OpBuilder else_bldr =
            OpBuilder::atBlockBegin(&if_outs.getElseRegion().front());
        Value partial_tensor = ktdf::ReadFromFifoOp::create(
            else_bldr, loc, output_tensor_type, fifo_out);
        scf::YieldOp::create(else_bldr, loc, partial_tensor);
      }
      Value outs_val = if_outs.getResult(0);

      // Clone linalg.generic with updated ins/outs.
      IRMapping mapping;
      mapping.map(generic_op.getInputs().front(), in_tensor.getResult());
      mapping.map(generic_op.getOutputs().front(), outs_val);
      auto new_generic = cast<linalg::GenericOp>(
          stage_bldr.clone(*generic_op.getOperation(), mapping));

      // Write result to fifo_out.
      ktdf::WriteToFifoOp::create(stage_bldr, loc, new_generic.getResult(0),
                                  fifo_out);
    }

    // -------- Store stage --------
    // Always writes fifo_out → partial_memref (which IS the L1 output buffer).
    auto store_new = ktdf::StageOp::create(pipe_bldr, loc,
                                           /*depends_in=*/ValueRange{tok1},
                                           /*depends_out=*/ValueRange{});
    if (store_units) store_new.setApplicableUnitsAttr(store_units);
    {
      OpBuilder stage_bldr(store_new.getBody(), store_new.getBody()->end());
      Value l1_idx = emit_l1_idx(stage_bldr, loc, batch_iv);

      auto dest_memref_type = cast<MemRefType>(partial_memref.getType());
      int64_t dest_rank = dest_memref_type.getRank();
      AffineMap dest_id =
          AffineMap::getMultiDimIdentityMap(dest_rank, stage_bldr.getContext());
      SmallVector<Value> dest_indices;
      dest_indices.push_back(l1_idx);
      for (int64_t i = 1; i < dest_rank; ++i) dest_indices.push_back(inner_c0);
      SmallVector<int64_t> dest_sizes;
      dest_sizes.push_back(1);
      for (int64_t i = 1; i < dest_rank; ++i)
        dest_sizes.push_back(dest_memref_type.getDimSize(i));
      ktdf::DataTransferOp::create(
          stage_bldr, loc, fifo_out, AffineMap{}, ValueRange{},
          ArrayRef<int64_t>{
              static_cast<int64_t>(out_fifo_type.getNumElements())},
          partial_memref, dest_id, dest_indices, dest_sizes);
    }

    // ------------------------------------------------------------------
    // Step 5: Erase the original inner pipeline.
    // ------------------------------------------------------------------
    rewriter.eraseOp(inner_pipeline);

    // Suppress unused-variable warnings for constants we computed but the
    // new IR accesses via the captured values above.
    (void)c_num_chunks;
    (void)c_num_chunks_m1;

    return success();
  }
};

}  // namespace

auto mlir::ktdf::createReductionDimChunkingPass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionDimChunkingPass>();
}
