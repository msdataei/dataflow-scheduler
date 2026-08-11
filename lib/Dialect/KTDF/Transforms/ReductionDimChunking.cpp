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
// The transformation replaces the single inner ktdf.pipeline with nested
// scf.for loops — one per reduction dimension that has more than 1 chunk.
// Dimensions whose num_chunks==1 produce no loop (the single chunk covers
// the entire dimension).  Each innermost iteration contains one ktdf.pipeline
// (Load / SFU / Store stages).  First-vs-rest accumulation behaviour is
// selected at runtime via %is_first = (all active loop IVs == 0):
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
// Helper: locate the input memref fed into the load stage by tracing the
// linalg.generic input back through read_from_fifo to the data_transfer in
// the load stage whose destination is that fifo slot.
//
// generic.getInputs()[0]
//   → defined by ReadFromFifoOp (fifo_in slot)
//   → DataTransferOp in load_stage with destination == fifo_in slot
//   → getSource() is the input memref
//
// Returns {load_transfer, read_from_fifo}, or {nullptr, nullptr} on failure.
// ---------------------------------------------------------------------------
static std::pair<ktdf::DataTransferOp, ktdf::ReadFromFifoOp> findLoadTransfer(
    ktdf::StageOp load_stage, linalg::GenericOp generic_op) {
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

// ---------------------------------------------------------------------------
// Helper: locate the partial-accumulation memref in the store stage by
// tracing the linalg.generic result through write_to_fifo to the
// data_transfer in the store stage whose source is that fifo slot.
//
// generic.getResult(0)
//   → consumed by WriteToFifoOp (fifo_out slot)
//   → DataTransferOp in store_stage with source == fifo_out slot
//   → getDestination() is the partial/output memref
//
// Returns {store_transfer, write_to_fifo}, or {nullptr, nullptr} on failure.
// ---------------------------------------------------------------------------
static std::pair<ktdf::DataTransferOp, ktdf::WriteToFifoOp> findStoreTransfer(
    ktdf::StageOp compute_stage, ktdf::StageOp store_stage,
    linalg::GenericOp generic_op) {
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

      // We only need loop_num_chunks for the debug log; actual loops are
      // per-dim so just track it for that purpose.
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
            << " total_chunks=" << loop_num_chunks;

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
                               chunk_sizes, per_dim_num_chunks);
  }

  // -----------------------------------------------------------------------
  // Core rewrite: replace the inner pipeline with nested scf.for loops —
  // one per reduction dimension that has num_chunks > 1.  Dimensions with
  // num_chunks == 1 produce no loop at all; the single chunk covers the
  // whole dimension and no iteration is required.
  //
  // For N active (chunked) dims the structure is:
  //
  //   scf.for %iv_0 = 0 to num_chunks[0] {     // only if num_chunks[0] > 1
  //     scf.for %iv_1 = 0 to num_chunks[1] {   // only if num_chunks[1] > 1
  //       ...
  //         %is_first = (iv_0 == 0 && iv_1 == 0 && ...)
  //         ktdf.pipeline { Load / SFU / Store }
  //     }
  //   }
  //
  // For dims whose num_chunks == 1, the IV is treated as the constant 0 when
  // computing offsets and is_first.
  // -----------------------------------------------------------------------
  LogicalResult rewriteComputeStage(
      ktdf::PipelineOp inner_pipeline, ktdf::StageOp load_stage,
      ktdf::StageOp compute_stage, ktdf::StageOp store_stage,
      linalg::GenericOp generic_op, ArrayRef<int64_t> reduction_dims,
      ArrayRef<int64_t> chunk_sizes, ArrayRef<int64_t> per_dim_num_chunks) {
    MLIRContext* context = inner_pipeline.getContext();
    IRRewriter rewriter(context);
    Location loc = inner_pipeline.getLoc();

    // ------------------------------------------------------------------
    // Step 1: Trace dataflow from linalg.generic ins/outs to find the
    // load/store data_transfer ops and the FIFO slot ops they connect to.
    //
    //   generic ins[0] ← ReadFromFifoOp(fifo_in) ←
    //   DataTransferOp(src=input_memref) generic result →
    //   WriteToFifoOp(fifo_out)  → DataTransferOp(dst=partial_memref)
    // ------------------------------------------------------------------
    auto [load_transfer, read_from_fifo] =
        findLoadTransfer(load_stage, generic_op);
    if (!read_from_fifo) {
      inner_pipeline.emitError(
          PASS_NAME ": generic input is not produced by read_from_fifo");
      return failure();
    }
    if (!load_transfer) {
      inner_pipeline.emitError(PASS_NAME
                               ": no memref-to-fifo transfer feeding "
                               "the generic's fifo_in slot in load stage");
      return failure();
    }
    Value input_memref = load_transfer.getSource();

    auto [store_transfer, write_to_fifo] =
        findStoreTransfer(compute_stage, store_stage, generic_op);
    if (!write_to_fifo) {
      inner_pipeline.emitError(
          PASS_NAME ": generic result is not consumed by write_to_fifo");
      return failure();
    }
    if (!store_transfer) {
      inner_pipeline.emitError(PASS_NAME
                               ": no fifo-to-memref transfer consuming "
                               "the generic's fifo_out slot in store stage");
      return failure();
    }
    Value partial_memref = store_transfer.getDestination();
    auto output_memref_type = cast<MemRefType>(partial_memref.getType());

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
    // Step 3: Build the replacement: nested scf.for loops (one per reduction
    // dim with num_chunks > 1) containing one ktdf.pipeline per innermost
    // iteration.  First-vs-rest behaviour is selected at runtime with scf.if
    // guards.
    //
    // We insert the outermost loop BEFORE inner_pipeline, then erase it.
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
    scf::ForOp batch_for =
        dyn_cast<scf::ForOp>(inner_pipeline->getParentRegion()->getParentOp());

    // Shared c0 / c1 constants emitted just before the outermost chunk loop.
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Per-dim chunk_size Value constants (emitted before the outermost loop).
    SmallVector<Value> c_chunk_sizes;
    for (int64_t cs : chunk_sizes)
      c_chunk_sizes.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, cs));

    // Per-dim num_chunks Value constants (emitted before the outermost loop).
    SmallVector<Value> c_per_dim_num_chunks;
    for (int64_t nc : per_dim_num_chunks)
      c_per_dim_num_chunks.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, nc));

    // Emit L1-slot index: divsi(subi(batch_iv, c0), c1).
    // Uses c0/c1 defined above.
    Value batch_iv = batch_for ? batch_for.getInductionVar() : Value{};
    auto emit_l1_idx = [&](OpBuilder& builder, Location l,
                           Value batch_iv_val) -> Value {
      Value sub = arith::SubIOp::create(builder, l, batch_iv_val, c0);
      return arith::DivSIOp::create(builder, l, sub, c1);
    };

    // ------------------------------------------------------------------
    // Build nested scf.for loops.  For each reduction dim j:
    //   - if per_dim_num_chunks[j] == 1: no loop; iv[j] = c0 (constant 0).
    //   - else: emit scf.for %iv_j = 0 to per_dim_num_chunks[j] step 1.
    // Track active IVs for is_first computation and the OpBuilder to use
    // when emitting the pipeline (starts at the innermost loop body or at
    // the current insertion point if no loops were generated).
    // ------------------------------------------------------------------
    size_t n_dims = reduction_dims.size();

    // Loop IVs per dim: either the loop's IV (if a loop was created for that
    // dim) or c0 (if num_chunks == 1 for that dim).
    SmallVector<Value> dim_ivs(n_dims);

    // We build loops from outermost (dim 0) to innermost (dim n_dims-1).
    // After the loop-building pass, `inner_builder` points to the body of
    // the innermost loop (or to the original insertion point when no loops
    // were generated).
    OpBuilder inner_builder = rewriter;  // copy: same insertion point

    for (size_t j = 0; j < n_dims; ++j) {
      if (per_dim_num_chunks[j] == 1) {
        // No loop for this dim: treat IV as constant 0.
        dim_ivs[j] = c0;
      } else {
        // Emit a scf.for for this dim.
        auto loop = scf::ForOp::create(inner_builder, loc, c0,
                                       c_per_dim_num_chunks[j], c1);
        dim_ivs[j] = loop.getInductionVar();
        // Move the builder into the loop body for the next (inner) level.
        inner_builder = OpBuilder(loop.getBody(), loop.getBody()->begin());
      }
    }

    // ------------------------------------------------------------------
    // Now emit the pipeline body using `inner_builder` (innermost context).
    // ------------------------------------------------------------------

    // %is_first = AND of (iv_j == 0) for all dims.
    // Dims with num_chunks==1 have iv==c0, so their cmpi is trivially true
    // but we omit them to avoid dead constants.  For dims with a real loop
    // IV we emit the compare and AND them together.
    Value is_first;
    for (size_t j = 0; j < n_dims; ++j) {
      if (per_dim_num_chunks[j] == 1) continue;  // iv is always 0
      Value eq = arith::CmpIOp::create(
          inner_builder, loc, arith::CmpIPredicate::eq, dim_ivs[j], c0);
      is_first = is_first
                     ? arith::AndIOp::create(inner_builder, loc, is_first, eq)
                     : eq;
    }
    // If every dim had num_chunks==1 (degenerate: pass shouldn't have been
    // called, but handle gracefully), is_first remains null — use c0 != c0
    // equivalent: always true.
    if (!is_first) {
      is_first = arith::CmpIOp::create(inner_builder, loc,
                                       arith::CmpIPredicate::eq, c0, c0);
    }

    // ---- Single pipeline per innermost chunk iteration ----
    auto phase_pipeline = ktdf::PipelineOp::create(inner_builder, loc);
    OpBuilder pipe_bldr(phase_pipeline.getBody(),
                        phase_pipeline.getBody()->end());

    // Private: fifo_in, fifo_out, tok0, tok1.
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

      // Per-reduction-dim row offsets: iv_j * chunk_size[j].
      SmallVector<Value> row_offsets;
      for (size_t j = 0; j < n_dims; ++j) {
        row_offsets.push_back(arith::MulIOp::create(stage_bldr, loc, dim_ivs[j],
                                                    c_chunk_sizes[j]));
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
          src_indices.push_back(c0);
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
      {
        OpBuilder else_bldr =
            OpBuilder::atBlockBegin(&if_op.getElseRegion().front());
        int64_t partial_rank = output_memref_type.getRank();
        AffineMap partial_id = AffineMap::getMultiDimIdentityMap(
            partial_rank, else_bldr.getContext());
        SmallVector<Value> partial_src_indices;
        partial_src_indices.push_back(l1_idx);
        for (int64_t i = 1; i < partial_rank; ++i)
          partial_src_indices.push_back(c0);
        SmallVector<int64_t> partial_src_sizes;
        partial_src_sizes.push_back(1);
        for (int64_t i = 1; i < partial_rank; ++i)
          partial_src_sizes.push_back(output_memref_type.getDimSize(i));
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

      auto in_tensor = ktdf::ReadFromFifoOp::create(
          stage_bldr, loc, chunk_input_tensor_type, fifo_in);

      auto if_outs = scf::IfOp::create(stage_bldr, loc,
                                       TypeRange{output_tensor_type}, is_first,
                                       /*withElseRegion=*/true);
      {
        OpBuilder then_bldr =
            OpBuilder::atBlockBegin(&if_outs.getThenRegion().front());
        Value empty_tensor = tensor::EmptyOp::create(
            then_bldr, loc, output_tensor_type.getShape(),
            output_tensor_type.getElementType());
        scf::YieldOp::create(then_bldr, loc, empty_tensor);
      }
      {
        OpBuilder else_bldr =
            OpBuilder::atBlockBegin(&if_outs.getElseRegion().front());
        Value partial_tensor = ktdf::ReadFromFifoOp::create(
            else_bldr, loc, output_tensor_type, fifo_out);
        scf::YieldOp::create(else_bldr, loc, partial_tensor);
      }
      Value outs_val = if_outs.getResult(0);

      IRMapping mapping;
      mapping.map(generic_op.getInputs().front(), in_tensor.getResult());
      mapping.map(generic_op.getOutputs().front(), outs_val);
      auto new_generic = cast<linalg::GenericOp>(
          stage_bldr.clone(*generic_op.getOperation(), mapping));

      ktdf::WriteToFifoOp::create(stage_bldr, loc, new_generic.getResult(0),
                                  fifo_out);
    }

    // -------- Store stage --------
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
      for (int64_t i = 1; i < dest_rank; ++i) dest_indices.push_back(c0);
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
    // Step 4: Erase the original inner pipeline.
    // ------------------------------------------------------------------
    rewriter.eraseOp(inner_pipeline);

    return success();
  }
};

}  // namespace

auto mlir::ktdf::createReductionDimChunkingPass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionDimChunkingPass>();
}

auto mlir::ktdf::createReductionDimChunkingPass(
    ReductionDimChunkingPassOptions options) -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionDimChunkingPass>(std::move(options));
}
