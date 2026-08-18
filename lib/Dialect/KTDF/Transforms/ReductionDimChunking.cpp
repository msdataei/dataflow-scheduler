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
//                         stage-a (L1LU)   : L1 → FIFO
//                         stage-b (compute): linalg.generic reduce
//                         stage-c (L1SU)   : FIFO → L1
//   stage-2 (MNISU)  : L1 → DDR store
//
// The transformation replaces the single inner ktdf.pipeline with nested
// scf.for loops — one per reduction dimension that has more than 1 chunk.
// Dimensions whose num_chunks==1 produce no loop (the single chunk covers
// the entire dimension).  Each innermost iteration contains one ktdf.pipeline
// (Load / Compute / Store stages).  First-vs-rest accumulation behaviour is
// selected at runtime via %condition = (all active loop IVs == 0):
//
//   Load stage   : always transfers the input chunk slice to fifo_in.
//                  When !condition, additionally transfers the partial
//                  accumulator (the existing L1 output buffer) to fifo_partial
//                  so the compute stage can read it back.
//   Compute stage: when condition, initialises the output tensor with
//                  tensor.empty; otherwise reads the partial result from
//                  fifo_partial.  The linalg.generic and write_to_fifo are
//                  unconditional.
//   Store stage  : unconditionally writes fifo_out back to the L1 output
//                  buffer.
//
// The existing L1 output buffer (discovered via the store stage's
// data_transfer destination) is reused as the partial-accumulation buffer
// — no new memref is allocated.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/ReductionChunkAnalysis.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
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
// Result of buildFifoPrivate: named accessors into the PrivateOp results so
// callers don't need to remember magic result-number offsets.
// ---------------------------------------------------------------------------
struct FifoPrivateResult {
  ktdf::PrivateOp priv_op;
  // Contiguous result slices in yield order: in-slots, partial-slots,
  // out-slots, tokens.
  unsigned in_begin;
  unsigned partial_begin;
  unsigned out_begin;
  unsigned tok_begin;

  Value inSlot(unsigned i) { return priv_op.getResult(in_begin + i); }
  Value partialSlot(unsigned i) { return priv_op.getResult(partial_begin + i); }
  Value outSlot(unsigned i) { return priv_op.getResult(out_begin + i); }
  Value token(unsigned i) { return priv_op.getResult(tok_begin + i); }
};

// ---------------------------------------------------------------------------
// Helper: build a ktdf.private yielding FIFO slots and tokens in the order:
//   [in_slots..., partial_slots..., out_slots..., tokens...]
//
// Counts determine how many allocations/tokens of each kind to create.
// All in-slots share `in_slot_type`, all partial-slots share
// `partial_slot_type`, all out-slots share `out_slot_type`.
// `partial_slot_type` is only consulted when n_partial_slots > 0.
// ---------------------------------------------------------------------------
static FifoPrivateResult buildFifoPrivate(
    OpBuilder& builder, Location loc, unsigned n_in_slots,
    ktdf::FifoSlotType in_slot_type, unsigned n_partial_slots,
    ktdf::FifoSlotType partial_slot_type, unsigned n_out_slots,
    ktdf::FifoSlotType out_slot_type, unsigned n_tokens) {
  auto tok_type = builder.getType<ktdf::TokenType>();

  SmallVector<Type> result_types;
  for (unsigned i = 0; i < n_in_slots; ++i)
    result_types.push_back(in_slot_type);
  for (unsigned i = 0; i < n_partial_slots; ++i)
    result_types.push_back(partial_slot_type);
  for (unsigned i = 0; i < n_out_slots; ++i)
    result_types.push_back(out_slot_type);
  for (unsigned i = 0; i < n_tokens; ++i) result_types.push_back(tok_type);

  auto priv_op = ktdf::PrivateOp::create(builder, loc, result_types);
  OpBuilder body_bldr(priv_op.getRegion());

  SmallVector<Value> yield_vals;
  auto alloc = [&](ktdf::FifoSlotType t) {
    return ktdf::FifoAllocateOp::create(body_bldr, loc, TypeRange{t},
                                        ValueRange{})
        .getResult(0);
  };
  for (unsigned i = 0; i < n_in_slots; ++i)
    yield_vals.push_back(alloc(in_slot_type));
  for (unsigned i = 0; i < n_partial_slots; ++i)
    yield_vals.push_back(alloc(partial_slot_type));
  for (unsigned i = 0; i < n_out_slots; ++i)
    yield_vals.push_back(alloc(out_slot_type));
  for (unsigned i = 0; i < n_tokens; ++i)
    yield_vals.push_back(
        ktdf::CreateTokenOp::create(body_bldr, loc, tok_type).getResult());

  ktdf::PrivateYieldOp::create(body_bldr, loc, yield_vals);

  FifoPrivateResult result;
  result.priv_op = priv_op;
  result.in_begin = 0;
  result.partial_begin = n_in_slots;
  result.out_begin = n_in_slots + n_partial_slots;
  result.tok_begin = n_in_slots + n_partial_slots + n_out_slots;
  return result;
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
  // find the immediately enclosing ktdf.stage (the compute stage) and collect
  // unique stages before transforming to avoid mutating the IR while walking.
  // -----------------------------------------------------------------------
  LogicalResult transformModule(ModuleOp module) {
    // Use a DenseSet to avoid O(n²) duplicate checks.
    llvm::DenseSet<ktdf::StageOp> seen;
    SmallVector<ktdf::StageOp> compute_stages;
    module.walk([&](linalg::GenericOp generic) {
      // Check if this generic has a reduction iterator.
      bool has_reduction = false;
      for (auto it : generic.getIteratorTypesArray()) {
        if (it == utils::IteratorType::reduction) {
          has_reduction = true;
          break;
        }
      }
      if (!has_reduction) return WalkResult::advance();

      // linalg.generic must always be directly nested inside a ktdf.stage.
      auto stage = generic->getParentOfType<ktdf::StageOp>();
      assert(stage && "linalg.generic not inside a ktdf.stage");

      if (seen.insert(stage).second) compute_stages.push_back(stage);
      return WalkResult::advance();
    });

    for (auto compute_stage : compute_stages)
      if (failed(transformPipeline(compute_stage))) return failure();

    return success();
  }

  // -----------------------------------------------------------------------
  // Transform the inner pipeline given the compute stage (the ktdf.stage
  // that directly contains the reduction linalg.generic).
  // -----------------------------------------------------------------------
  LogicalResult transformPipeline(ktdf::StageOp compute_stage) {
    // Stages are always direct children of a pipeline.
    auto inner_pipeline = cast<ktdf::PipelineOp>(compute_stage->getParentOp());

    // Verify there is exactly one inner pipeline inside the parent region.
    unsigned inner_pipeline_count = 0;
    inner_pipeline->getParentRegion()->walk<WalkOrder::PreOrder>(
        [&](ktdf::PipelineOp) {
          ++inner_pipeline_count;
          return WalkResult::skip();
        });
    assert(inner_pipeline_count == 1 &&
           "expected exactly 1 inner pipeline in parent region");

    ktdf::StageOp load_stage = findLoadStage(inner_pipeline, compute_stage);
    assert(load_stage && "could not find load stage upstream of compute stage");

    ktdf::StageOp store_stage = findStoreStage(inner_pipeline, compute_stage);
    assert(store_stage &&
           "could not find store stage downstream of compute stage");

    // Locate the linalg.generic with a reduction iterator.
    linalg::GenericOp generic_op = findReductionGenericOp(compute_stage);
    assert(generic_op && "no reduction linalg.generic in compute stage");

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
        inner_pipeline.emitError(PASS_NAME
                                 ": ReductionChunkAnalysis could not infer "
                                 "chunk count");
        return failure();
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
        inner_pipeline.emitError(PASS_NAME
                                 ": could not find reduction dimension");
        return failure();
      }

      if (numChunks.size() != reduction_dims.size()) {
        inner_pipeline.emitError(
            llvm::Twine(PASS_NAME ": numChunks has ") +
            llvm::Twine(numChunks.size()) + " entries but there are " +
            llvm::Twine(reduction_dims.size()) + " reduction dims");
        return failure();
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
  //   scf.for %iv_0 = 0 to num_chunks[0] {       // only if num_chunks[0] > 1
  //     scf.for %iv_1 = 0 to num_chunks[1] {     // only if num_chunks[1] > 1
  //       ...
  //         %condition = (iv_0 == 0 && iv_1 == 0 && ...)
  //         ktdf.pipeline { Load / Compute / Store }
  //     }
  //   }
  //
  // For dims whose num_chunks == 1, the IV is treated as the constant 0 when
  // computing offsets and condition.
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
    //   generic.ins[0]    ← ReadFromFifoOp(fifo_in)
    //                     ← DataTransferOp(src=input_memref, dst=fifo_in)
    //   generic.result(0) → WriteToFifoOp(fifo_out)
    //                     → DataTransferOp(src=fifo_out, dst=partial_memref)
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
    // Step 2: Build the replacement: nested scf.for loops (one per reduction
    // dim with num_chunks > 1) containing one ktdf.pipeline per innermost
    // iteration.  First-vs-rest behaviour is selected at runtime via scf.if
    // guards on %condition.
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
    auto compute_units = compute_stage.getApplicableUnitsAttr();
    auto store_units = store_stage.getApplicableUnitsAttr();

    // Batch-loop induction variable.
    auto batch_for =
        cast<scf::ForOp>(inner_pipeline->getParentRegion()->getParentOp());
    assert(batch_for && "inner pipeline not nested inside a batch scf.for");

    // Shared c0 / c1 constants emitted just before the outermost chunk loop.
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Per-dim num_chunks Value constants (emitted before the outermost loop).
    SmallVector<Value> c_per_dim_num_chunks;
    for (int64_t nc : per_dim_num_chunks)
      c_per_dim_num_chunks.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, nc));

    // L1-slot index is the batch induction variable itself; c0/c1 are used
    // when building the index arithmetic inside each stage body.
    Value batch_iv = batch_for.getInductionVar();
    auto emit_l1_idx = [&](OpBuilder& builder, Location l,
                           Value batch_iv_val) -> Value {
      Value sub = arith::SubIOp::create(builder, l, batch_iv_val, c0);
      return arith::DivSIOp::create(builder, l, sub, c1);
    };

    // ------------------------------------------------------------------
    // Build nested scf.for loops.  For each reduction dim j:
    //   - if per_dim_num_chunks[j] == 1: no loop; iv[j] = c0 (constant 0).
    //   - else: emit scf.for %iv_j = 0 to per_dim_num_chunks[j] step 1.
    // Track active IVs for condition computation and the OpBuilder to use
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

    // condition = AND of (iv_j == 0) for all dims with num_chunks > 1.
    // Dims with num_chunks==1 have iv==c0, so their cmpi is trivially true
    // but we omit them to avoid dead constants.
    Value condition;
    for (size_t j = 0; j < n_dims; ++j) {
      if (per_dim_num_chunks[j] == 1) continue;  // iv is always 0
      Value eq = arith::CmpIOp::create(
          inner_builder, loc, arith::CmpIPredicate::eq, dim_ivs[j], c0);
      condition = condition
                      ? arith::AndIOp::create(inner_builder, loc, condition, eq)
                      : eq;
    }
    // If every dim had num_chunks==1 (degenerate: pass shouldn't have been
    // called, but handle gracefully), condition remains null — emit
    // arith.constant true directly rather than a redundant cmpi.
    if (!condition) {
      condition = arith::ConstantIntOp::create(inner_builder, loc, /*value=*/1,
                                               /*width=*/1);
    }

    // ---- Single pipeline per innermost chunk iteration ----
    auto phase_pipeline = ktdf::PipelineOp::create(inner_builder, loc);
    OpBuilder pipe_bldr(phase_pipeline.getBody(),
                        phase_pipeline.getBody()->end());

    // Build a partial-slot FIFO type with the Load→Compute direction
    // (same src/dest as fifo_in) but carrying output-tensor-sized elements.
    auto partial_fifo_type = ktdf::FifoSlotType::get(
        context, in_fifo_type.getSrc(), in_fifo_type.getDest(),
        out_fifo_type.getNumElements(), out_fifo_type.getElementType());

    // Private: 1 in-slot, 1 partial-slot, 1 out-slot, 2 tokens.
    //   fifo_in      (in-slot)      : Load → Compute  input chunk
    //   fifo_partial (partial-slot) : Load → Compute  partial accumulator
    //                                 (non-first iterations only)
    //   fifo_out     (out-slot)     : Compute → Store reduction result
    auto slots = buildFifoPrivate(pipe_bldr, loc,
                                  /*n_in_slots=*/1, chunk_in_fifo_type,
                                  /*n_partial_slots=*/1, partial_fifo_type,
                                  /*n_out_slots=*/1, out_fifo_type,
                                  /*n_tokens=*/2);
    Value fifo_in = slots.inSlot(0);
    Value fifo_partial = slots.partialSlot(0);
    Value fifo_out = slots.outSlot(0);
    Value tok0 = slots.token(0);
    Value tok1 = slots.token(1);

    buildLoadStage(pipe_bldr, loc, tok0, load_units, condition, batch_iv,
                   input_memref, input_memref_type, partial_memref,
                   output_memref_type, partial_fifo_type, fifo_in, fifo_partial,
                   chunk_fifo_elements, reduction_dims, chunk_sizes, dim_ivs,
                   c0, emit_l1_idx);

    buildComputeStage(pipe_bldr, loc, tok0, tok1, compute_units, condition,
                      fifo_in, fifo_partial, fifo_out, chunk_input_tensor_type,
                      output_tensor_type, generic_op);

    buildStoreStage(pipe_bldr, loc, tok1, store_units, batch_iv, partial_memref,
                    out_fifo_type, fifo_out, c0, emit_l1_idx);

    // ------------------------------------------------------------------
    // Step 3: Erase the original inner pipeline.
    // ------------------------------------------------------------------
    rewriter.eraseOp(inner_pipeline);

    return success();
  }

  // -----------------------------------------------------------------------
  // Helper: build the load stage inside a chunk pipeline.
  //
  // The load stage always transfers one input chunk slice (memref → fifo_in).
  // On non-first iterations (!condition) it also feeds the existing partial
  // accumulator (partial_memref → fifo_partial) so the compute stage can read
  // it.
  //
  // Input memref layout: dim-0 is the batch/L1 slot index; dims 1..rank-1
  // map to linalg.generic iterator dims 0..rank-2.  For each memref dim i
  // (i ≥ 1), the corresponding generic dim is (i-1).  If that generic dim is
  // a reduction dim, the source offset is iv_j * chunk_size[j] and the static
  // size is chunk_size[j]; otherwise offset is 0 and size is the full dim.
  // -----------------------------------------------------------------------
  void buildLoadStage(
      OpBuilder& pipe_bldr, Location loc, Value tok0, ArrayAttr load_units,
      Value condition, Value batch_iv, Value input_memref,
      MemRefType input_memref_type, Value partial_memref,
      MemRefType output_memref_type, ktdf::FifoSlotType partial_fifo_type,
      Value fifo_in, Value fifo_partial, int64_t chunk_fifo_elements,
      ArrayRef<int64_t> reduction_dims, ArrayRef<int64_t> chunk_sizes,
      ArrayRef<Value> dim_ivs, Value c0,
      llvm::function_ref<Value(OpBuilder&, Location, Value)> emit_l1_idx) {
    auto stage = ktdf::StageOp::create(pipe_bldr, loc,
                                       /*depends_in=*/ValueRange{},
                                       /*depends_out=*/ValueRange{tok0});
    if (load_units) stage.setApplicableUnitsAttr(load_units);

    OpBuilder b(stage.getBody(), stage.getBody()->end());
    Value l1_idx = emit_l1_idx(b, loc, batch_iv);

    // Build source indices, static sizes, and an AffineMap for the
    // data_transfer together.
    //
    // dim-0 of the memref is the batch/L1 slot (index = l1_idx, size = 1).
    // dims 1..rank-1 map to generic dims 0..rank-2:
    //   - reduction dim j → index operand = dim_ivs[j],
    //                        map result    = d_j * chunk_size[j]  (affine),
    //                        static size   = chunk_size[j]
    //   - parallel dim    → index operand = c0,
    //                        map result    = d_k  (identity),
    //                        static size   = full dim size
    //
    // Encoding iv * chunk_size inside the AffineMap avoids emitting an
    // arith.muli SSA value as an index operand, which the backend does not
    // support.
    int64_t input_rank = input_memref_type.getRank();
    SmallVector<Value> src_indices;
    SmallVector<int64_t> src_static_sizes;
    SmallVector<AffineExpr> map_results;
    src_indices.reserve(static_cast<size_t>(input_rank));
    src_static_sizes.reserve(static_cast<size_t>(input_rank));
    map_results.reserve(static_cast<size_t>(input_rank));

    // dim-0: batch/L1 slot.
    src_indices.push_back(l1_idx);
    src_static_sizes.push_back(1);
    map_results.push_back(getAffineDimExpr(0, b.getContext()));

    // dims 1..rank-1.
    for (int64_t i = 1; i < input_rank; ++i) {
      int64_t generic_dim = i - 1;
      auto it = llvm::find(reduction_dims, generic_dim);
      unsigned dim_idx = static_cast<unsigned>(i);
      if (it != reduction_dims.end()) {
        size_t j = static_cast<size_t>(it - reduction_dims.begin());
        src_indices.push_back(dim_ivs[j]);
        src_static_sizes.push_back(chunk_sizes[j]);
        // iv_j * chunk_size[j]: constant multiplication stays in the map.
        map_results.push_back(getAffineDimExpr(dim_idx, b.getContext()) *
                              chunk_sizes[j]);
      } else {
        src_indices.push_back(c0);
        src_static_sizes.push_back(input_memref_type.getDimSize(i));
        map_results.push_back(getAffineDimExpr(dim_idx, b.getContext()));
      }
    }

    AffineMap input_map =
        AffineMap::get(static_cast<unsigned>(input_rank),
                       /*symbolCount=*/0, map_results, b.getContext());
    ktdf::DataTransferOp::create(
        b, loc, input_memref, input_map, src_indices, src_static_sizes, fifo_in,
        AffineMap{}, ValueRange{}, ArrayRef<int64_t>{chunk_fifo_elements});

    // On non-first iterations, feed the partial accumulator into fifo_partial
    // so the compute stage can read it back as the running result.
    auto if_op =
        scf::IfOp::create(b, loc, /*resultTypes=*/TypeRange{}, condition,
                          /*withElseRegion=*/true);
    {
      OpBuilder else_bldr =
          OpBuilder::atBlockBegin(&if_op.getElseRegion().front());
      int64_t partial_rank = output_memref_type.getRank();
      AffineMap partial_id = AffineMap::getMultiDimIdentityMap(
          partial_rank, else_bldr.getContext());
      SmallVector<Value> partial_src_indices;
      SmallVector<int64_t> partial_src_sizes;
      partial_src_indices.reserve(static_cast<size_t>(partial_rank));
      partial_src_sizes.reserve(static_cast<size_t>(partial_rank));
      partial_src_indices.push_back(l1_idx);
      partial_src_sizes.push_back(1);
      for (int64_t i = 1; i < partial_rank; ++i) {
        partial_src_indices.push_back(c0);
        partial_src_sizes.push_back(output_memref_type.getDimSize(i));
      }
      ktdf::DataTransferOp::create(
          else_bldr, loc, partial_memref, partial_id, partial_src_indices,
          partial_src_sizes, fifo_partial, AffineMap{}, ValueRange{},
          ArrayRef<int64_t>{
              static_cast<int64_t>(partial_fifo_type.getNumElements())});
    }
  }

  // -----------------------------------------------------------------------
  // Helper: build the compute stage inside a chunk pipeline.
  //
  // On the first iteration (condition == true) the output is initialised with
  // tensor.empty.  On subsequent iterations the partial result is read from
  // fifo_partial.  The cloned linalg.generic and the write_to_fifo are
  // unconditional.
  // -----------------------------------------------------------------------
  void buildComputeStage(OpBuilder& pipe_bldr, Location loc, Value tok0,
                         Value tok1, ArrayAttr compute_units, Value condition,
                         Value fifo_in, Value fifo_partial, Value fifo_out,
                         RankedTensorType chunk_input_tensor_type,
                         RankedTensorType output_tensor_type,
                         linalg::GenericOp generic_op) {
    auto stage = ktdf::StageOp::create(pipe_bldr, loc,
                                       /*depends_in=*/ValueRange{tok0},
                                       /*depends_out=*/ValueRange{tok1});
    if (compute_units) stage.setApplicableUnitsAttr(compute_units);

    OpBuilder b(stage.getBody(), stage.getBody()->end());

    auto in_tensor =
        ktdf::ReadFromFifoOp::create(b, loc, chunk_input_tensor_type, fifo_in);

    // Select the output tensor: tensor.empty on first iteration, or the
    // partial result read from fifo_partial on subsequent iterations.
    auto if_op = scf::IfOp::create(b, loc, TypeRange{output_tensor_type},
                                   condition, /*withElseRegion=*/true);
    {
      OpBuilder then_bldr =
          OpBuilder::atBlockBegin(&if_op.getThenRegion().front());
      Value empty_tensor =
          tensor::EmptyOp::create(then_bldr, loc, output_tensor_type.getShape(),
                                  output_tensor_type.getElementType());
      scf::YieldOp::create(then_bldr, loc, empty_tensor);
    }
    {
      OpBuilder else_bldr =
          OpBuilder::atBlockBegin(&if_op.getElseRegion().front());
      Value partial_tensor = ktdf::ReadFromFifoOp::create(
          else_bldr, loc, output_tensor_type, fifo_partial);
      scf::YieldOp::create(else_bldr, loc, partial_tensor);
    }

    IRMapping mapping;
    mapping.map(generic_op.getInputs().front(), in_tensor.getResult());
    mapping.map(generic_op.getOutputs().front(), if_op.getResult(0));
    auto new_generic =
        cast<linalg::GenericOp>(b.clone(*generic_op.getOperation(), mapping));

    ktdf::WriteToFifoOp::create(b, loc, new_generic.getResult(0), fifo_out);
  }

  // -----------------------------------------------------------------------
  // Helper: build the store stage inside a chunk pipeline.
  //
  // Unconditionally reads the result from fifo_out and writes it back to the
  // L1 partial-accumulation buffer.
  // -----------------------------------------------------------------------
  void buildStoreStage(
      OpBuilder& pipe_bldr, Location loc, Value tok1, ArrayAttr store_units,
      Value batch_iv, Value partial_memref, ktdf::FifoSlotType out_fifo_type,
      Value fifo_out, Value c0,
      llvm::function_ref<Value(OpBuilder&, Location, Value)> emit_l1_idx) {
    auto stage = ktdf::StageOp::create(pipe_bldr, loc,
                                       /*depends_in=*/ValueRange{tok1},
                                       /*depends_out=*/ValueRange{});
    if (store_units) stage.setApplicableUnitsAttr(store_units);

    OpBuilder b(stage.getBody(), stage.getBody()->end());
    Value l1_idx = emit_l1_idx(b, loc, batch_iv);

    auto dest_memref_type = cast<MemRefType>(partial_memref.getType());
    int64_t dest_rank = dest_memref_type.getRank();
    AffineMap dest_id =
        AffineMap::getMultiDimIdentityMap(dest_rank, b.getContext());
    SmallVector<Value> dest_indices;
    SmallVector<int64_t> dest_sizes;
    dest_indices.reserve(static_cast<size_t>(dest_rank));
    dest_sizes.reserve(static_cast<size_t>(dest_rank));
    dest_indices.push_back(l1_idx);
    dest_sizes.push_back(1);
    for (int64_t i = 1; i < dest_rank; ++i) {
      dest_indices.push_back(c0);
      dest_sizes.push_back(dest_memref_type.getDimSize(i));
    }
    ktdf::DataTransferOp::create(
        b, loc, fifo_out, AffineMap{}, ValueRange{},
        ArrayRef<int64_t>{static_cast<int64_t>(out_fifo_type.getNumElements())},
        partial_memref, dest_id, dest_indices, dest_sizes);
  }
};

}  // namespace

auto mlir::ktdf::createReductionDimChunkingPass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionDimChunkingPass>();
}
