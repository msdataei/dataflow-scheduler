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
//   stage-0 (Load)   : DDR → L1
//   stage-1 (Compute): outer scf.for (batch) whose body is a single inner
//                       ktdf.pipeline with three stages:
//                         stage-a (Load)   : L1 → FIFO
//                         stage-b (Compute): linalg.generic reduce
//                         stage-c (Store)  : FIFO → L1
//   stage-2 (Store)  : L1 → DDR
//
// The transformation replaces the single inner ktdf.pipeline with nested
// scf.for loops — one per reduction dimension that has more than one chunk.
// Dimensions whose num_chunks==1 produce no loop (the single chunk covers
// the entire dimension).  Each innermost iteration contains one ktdf.pipeline
// with three stages built by StageFactory (Load / Compute / Store).
// First-vs-rest accumulation behaviour is selected at runtime via
// %condition = (all active loop IVs == 0):
//
//   Load stage   : transfers each input chunk slice (memref → fifo_in[i]).
//                  When !condition, also transfers the partial accumulator
//                  (L1 output buffer → fifo_partial[i]) so the Compute stage
//                  can read it back.
//   Compute stage: when condition, initialises the output tensor with
//                  tensor.empty; otherwise reads the partial result from
//                  fifo_partial.  The linalg.generic and write_to_fifo are
//                  unconditional.
//   Store stage  : unconditionally writes each fifo_out[i] back to the L1
//                  output buffer.
//
// The existing L1 output buffer (discovered via the original Store stage's
// data_transfer destination) is reused as the partial-accumulation buffer;
// no new memref is allocated.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/ReductionChunkAnalysis.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "llvm/ADT/DenseSet.h"
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
// Result of StageFactory::buildFifoPrivate.
//
// Wraps the ktdf.private op and records the begin-offset of each group of
// results so callers can retrieve any slot or token by logical index without
// knowing the raw result-number layout.
//
// Yield order inside the ktdf.private body (and thus result order):
//   [in_slots... | partial_slots... | out_slots... | tokens...]
// ---------------------------------------------------------------------------
struct FifoPrivateResult {
  ktdf::PrivateOp priv_op;
  // Begin offset of each group in the PrivateOp result list.
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
// Describes the FIFO slot layout and token count for one chunk pipeline.
//
// Passed to StageFactory::buildFifoPrivate, buildLoadStage, buildComputeStage,
// and buildStoreStage so that explicit per-slot index arguments are no longer
// needed at every call site.
//
// Slot groups (parallel vectors; index i selects one logical slot within that
// group):
//   in_slot_types     — FifoSlotType for each input-chunk FIFO (Load → Compute)
//   partial_slot_types— FifoSlotType for each partial-accum FIFO (Load →
//   Compute,
//                       non-first iterations only); may be empty.
//   out_slot_types    — FifoSlotType for each result FIFO (Compute → Store)
//
// Token assignment (fixed convention used by all three stage builders):
//   token 0 … n_tokens-2 : Load → Compute synchronisation tokens
//   token n_tokens-1      : Compute → Store synchronisation token
//
// Currently the convention is n_tokens == 2: token 0 for Load→Compute and
// token 1 for Compute→Store.
// ---------------------------------------------------------------------------
struct ChunkPipelineConfig {
  SmallVector<ktdf::FifoSlotType> in_slot_types;
  SmallVector<ktdf::FifoSlotType> partial_slot_types;
  SmallVector<ktdf::FifoSlotType> out_slot_types;
  unsigned n_tokens = 0;
};

// ---------------------------------------------------------------------------
// StageFactory: builds the Load / Compute / Store stages of one chunk-pipeline
// iteration inside a ktdf.pipeline body.
//
// Usage sequence:
//   1. Construct with the pipeline body builder, location, batch IV, and
//      per-stage applicable-unit attributes.
//   2. Call buildFifoPrivate(cfg) to allocate FIFO slots and tokens described
//      by ChunkPipelineConfig cfg; it returns a FifoPrivateResult with named
//      accessors.
//   3. Call setSlots() with that result so the stage builders can address
//      any slot or token by index.
//   4. Call buildLoadStage(…, cfg), buildComputeStage(…, cfg),
//      buildStoreStage(…, cfg) in order.
//
// The ChunkPipelineConfig carries all slot-type lists and the token count, so
// no per-slot index arguments need to be passed to the stage builders.
//
// The L1-index arithmetic (batch_iv → L1 slot index) is encapsulated in
// emitL1Idx() and never needs to be passed as a parameter.
//
// This class can be reused whenever a 3-stage Load / Compute / Store pipeline
// is required.
// ---------------------------------------------------------------------------
class StageFactory {
 public:
  // -------------------------------------------------------------------------
  // Records the pipeline body builder, location, batch induction variable,
  // and per-stage applicable-unit attributes.  Does not emit any IR.
  // -------------------------------------------------------------------------
  StageFactory(OpBuilder& pipe_bldr, Location loc, Value batch_iv,
               ArrayAttr load_units, ArrayAttr compute_units,
               ArrayAttr store_units)
      : pipe_bldr_(pipe_bldr),
        loc_(loc),
        batch_iv_(batch_iv),
        load_units_(load_units),
        compute_units_(compute_units),
        store_units_(store_units) {}

  // -------------------------------------------------------------------------
  // Emit a ktdf.private op that yields FIFO slots and tokens in the order:
  //   [in_slots... | partial_slots... | out_slots... | tokens...]
  //
  // The layout is driven entirely by cfg:
  //   cfg.in_slot_types.size()      in-slots
  //   cfg.partial_slot_types.size() partial-slots (skipped if empty)
  //   cfg.out_slot_types.size()     out-slots
  //   cfg.n_tokens                  ktdf.token values
  //
  // Returns a FifoPrivateResult whose named accessors (inSlot, partialSlot,
  // outSlot, token) index into the corresponding group.  Call setSlots() with
  // the result before invoking any stage builder.
  // -------------------------------------------------------------------------
  FifoPrivateResult buildFifoPrivate(const ChunkPipelineConfig& cfg) {
    auto tok_type = pipe_bldr_.getType<ktdf::TokenType>();

    SmallVector<Type> result_types;
    for (auto slot_type : cfg.in_slot_types) {
      result_types.push_back(slot_type);
    }
    for (auto slot_type : cfg.partial_slot_types) {
      result_types.push_back(slot_type);
    }
    for (auto slot_type : cfg.out_slot_types) {
      result_types.push_back(slot_type);
    }
    for (unsigned i = 0; i < cfg.n_tokens; ++i) {
      result_types.push_back(tok_type);
    }

    auto priv_op = ktdf::PrivateOp::create(pipe_bldr_, loc_, result_types);
    OpBuilder body_bldr(priv_op.getRegion());

    SmallVector<Value> yield_vals;
    auto alloc = [&](ktdf::FifoSlotType slot_type) {
      return ktdf::FifoAllocateOp::create(body_bldr, loc_, TypeRange{slot_type},
                                          ValueRange{})
          .getResult(0);
    };
    for (auto slot_type : cfg.in_slot_types) {
      yield_vals.push_back(alloc(slot_type));
    }
    for (auto slot_type : cfg.partial_slot_types) {
      yield_vals.push_back(alloc(slot_type));
    }
    for (auto slot_type : cfg.out_slot_types) {
      yield_vals.push_back(alloc(slot_type));
    }
    for (unsigned i = 0; i < cfg.n_tokens; ++i) {
      yield_vals.push_back(
          ktdf::CreateTokenOp::create(body_bldr, loc_, tok_type).getResult());
    }

    ktdf::PrivateYieldOp::create(body_bldr, loc_, yield_vals);

    auto n_in = static_cast<unsigned>(cfg.in_slot_types.size());
    auto n_partial = static_cast<unsigned>(cfg.partial_slot_types.size());
    auto n_out = static_cast<unsigned>(cfg.out_slot_types.size());
    FifoPrivateResult result;
    result.priv_op = priv_op;
    result.in_begin = 0;
    result.partial_begin = n_in;
    result.out_begin = n_in + n_partial;
    result.tok_begin = n_in + n_partial + n_out;
    return result;
  }

  // Bind the FifoPrivateResult produced by buildFifoPrivate() into the
  // factory.  Must be called before any of the stage builders.
  void setSlots(FifoPrivateResult slots) { slots_ = slots; }

  // -------------------------------------------------------------------------
  // Emit the Load stage inside a chunk pipeline.
  //
  // For each slot in cfg.in_slot_types: transfers one input chunk slice from
  // input_memref into the corresponding in-slot (input_memref → fifo_in[i]).
  //
  // For each slot in cfg.partial_slot_types: on non-first iterations
  // (!condition) transfers the running partial accumulator from partial_memref
  // into the corresponding partial-slot (partial_memref → fifo_partial[i]).
  // If cfg.partial_slot_types is empty the scf.if guard is not emitted.
  //
  // Token assignment follows the ChunkPipelineConfig convention:
  //   token 0 : Load → Compute (depends_out).
  //
  // Parameters:
  //   condition          — i1 value; true on the first chunk iteration.
  //   input_memref       — source buffer for the input chunk.
  //   input_memref_type  — type of input_memref (cached to avoid re-cast).
  //   partial_memref     — L1 buffer reused as the partial accumulator.
  //   output_memref_type — type of partial_memref (cached to avoid re-cast).
  //   reduction_dims     — linalg.generic dims that are being chunked.
  //   chunk_sizes        — size of one chunk along each reduction dim.
  //   dim_ivs            — loop IVs (or c0) for each reduction dim.
  //   cfg                — slot layout and token count for this pipeline.
  //
  // Input memref layout: dim-0 is the batch/L1 slot index; dims 1..rank-1
  // correspond to linalg.generic dims 0..rank-2.  For reduction dims the
  // source offset is encoded as iv_j * chunk_size[j] in the AffineMap (to
  // avoid emitting arith.muli, which the backend does not accept as an index).
  // -------------------------------------------------------------------------
  void buildLoadStage(Value condition, Value input_memref,
                      MemRefType input_memref_type, Value partial_memref,
                      MemRefType output_memref_type,
                      ArrayRef<int64_t> reduction_dims,
                      ArrayRef<int64_t> chunk_sizes, ArrayRef<Value> dim_ivs,
                      const ChunkPipelineConfig& cfg) {
    // token 0: Load → Compute synchronisation (ChunkPipelineConfig convention).
    auto stage =
        ktdf::StageOp::create(pipe_bldr_, loc_,
                              /*depends_in=*/ValueRange{},
                              /*depends_out=*/ValueRange{slots_.token(0)});
    if (load_units_) {
      stage.setApplicableUnitsAttr(load_units_);
    }

    OpBuilder b(stage.getBody(), stage.getBody()->end());
    Value c0 = arith::ConstantIndexOp::create(b, loc_, 0);
    Value l1_idx = emitL1Idx(b, c0);

    // Build the shared source index vector and AffineMap for all in-slots.
    //
    // dim-0 of the memref is the batch/L1 slot (index = l1_idx, size = 1).
    // dims 1..rank-1 correspond to linalg.generic dims 0..rank-2:
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

    // Transfer the input chunk into each in-slot.
    for (unsigned i = 0; i < cfg.in_slot_types.size(); ++i) {
      Value fifo_in_slot = slots_.inSlot(i);
      auto fifo_type = cast<ktdf::FifoSlotType>(fifo_in_slot.getType());
      ktdf::DataTransferOp::create(
          b, loc_, input_memref, input_map, src_indices, src_static_sizes,
          fifo_in_slot, AffineMap{}, ValueRange{},
          ArrayRef<int64_t>{static_cast<int64_t>(fifo_type.getNumElements())});
    }

    // On non-first iterations, feed each partial accumulator into its partial-
    // slot so the Compute stage can read back the running result.
    if (!cfg.partial_slot_types.empty()) {
      auto if_op =
          scf::IfOp::create(b, loc_, /*resultTypes=*/TypeRange{}, condition,
                            /*withElseRegion=*/true);
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
      for (unsigned i = 0; i < cfg.partial_slot_types.size(); ++i) {
        Value fifo_partial_slot = slots_.partialSlot(i);
        auto fifo_type = cast<ktdf::FifoSlotType>(fifo_partial_slot.getType());
        ktdf::DataTransferOp::create(
            else_bldr, loc_, partial_memref, partial_id, partial_src_indices,
            partial_src_sizes, fifo_partial_slot, AffineMap{}, ValueRange{},
            ArrayRef<int64_t>{
                static_cast<int64_t>(fifo_type.getNumElements())});
      }
    }
  }

  // -------------------------------------------------------------------------
  // Emit the Compute stage inside a chunk pipeline.
  //
  // Reads the input chunk from fifo_in[0] (unconditional).
  // Selects the initial output tensor via a runtime branch on condition:
  //   - condition == true  (first iteration): tensor.empty
  //   - condition == false (subsequent)     : read from fifo_partial[0]
  // Runs the cloned linalg.generic, then writes the result to
  // fifo_out[0] (unconditional).
  //
  // Token assignment follows the ChunkPipelineConfig convention:
  //   token 0             : Load → Compute (depends_in).
  //   token n_tokens-1    : Compute → Store (depends_out).
  //
  // Parameters:
  //   condition               — i1 value; true on the first chunk iteration.
  //   chunk_input_tensor_type — tensor type for one input chunk.
  //   output_tensor_type      — tensor type for the reduction result.
  //   generic_op              — original linalg.generic to clone.
  //   cfg                     — slot layout and token count for this pipeline.
  // -------------------------------------------------------------------------
  void buildComputeStage(Value condition,
                         RankedTensorType chunk_input_tensor_type,
                         RankedTensorType output_tensor_type,
                         linalg::GenericOp generic_op,
                         const ChunkPipelineConfig& cfg) {
    // token 0:          Load → Compute (depends_in).
    // token n_tokens-1: Compute → Store (depends_out).
    auto stage = ktdf::StageOp::create(
        pipe_bldr_, loc_,
        /*depends_in=*/ValueRange{slots_.token(0)},
        /*depends_out=*/ValueRange{slots_.token(cfg.n_tokens - 1)});
    if (compute_units_) {
      stage.setApplicableUnitsAttr(compute_units_);
    }

    OpBuilder b(stage.getBody(), stage.getBody()->end());

    auto in_tensor = ktdf::ReadFromFifoOp::create(
        b, loc_, chunk_input_tensor_type, slots_.inSlot(0));

    // Select the initial output tensor for linalg.generic:
    //   first iteration  → tensor.empty (no prior partial result)
    //   later iterations → partial result read from fifo_partial[0]
    auto if_op = scf::IfOp::create(b, loc_, TypeRange{output_tensor_type},
                                   condition, /*withElseRegion=*/true);
    {
      OpBuilder then_bldr =
          OpBuilder::atBlockBegin(&if_op.getThenRegion().front());
      Value empty_tensor = tensor::EmptyOp::create(
          then_bldr, loc_, output_tensor_type.getShape(),
          output_tensor_type.getElementType());
      scf::YieldOp::create(then_bldr, loc_, empty_tensor);
    }
    {
      OpBuilder else_bldr =
          OpBuilder::atBlockBegin(&if_op.getElseRegion().front());
      Value partial_tensor = ktdf::ReadFromFifoOp::create(
          else_bldr, loc_, output_tensor_type, slots_.partialSlot(0));
      scf::YieldOp::create(else_bldr, loc_, partial_tensor);
    }

    IRMapping mapping;
    mapping.map(generic_op.getInputs().front(), in_tensor.getResult());
    mapping.map(generic_op.getOutputs().front(), if_op.getResult(0));
    auto new_generic =
        cast<linalg::GenericOp>(b.clone(*generic_op.getOperation(), mapping));

    ktdf::WriteToFifoOp::create(b, loc_, new_generic.getResult(0),
                                slots_.outSlot(0));
  }

  // -------------------------------------------------------------------------
  // Emit the Store stage inside a chunk pipeline.
  //
  // For each slot in cfg.out_slot_types: unconditionally reads the reduction
  // result from fifo_out[i] and writes it back to partial_memref (the L1
  // partial-accumulation buffer).
  //
  // Token assignment follows the ChunkPipelineConfig convention:
  //   token n_tokens-1 : Compute → Store (depends_in).
  //
  // Parameters:
  //   partial_memref — L1 destination buffer (same as the one the Load
  //                    stage reads the running partial result from).
  //   cfg            — slot layout and token count for this pipeline.
  // -------------------------------------------------------------------------
  void buildStoreStage(Value partial_memref, const ChunkPipelineConfig& cfg) {
    // token n_tokens-1: Compute → Store (ChunkPipelineConfig convention).
    auto stage = ktdf::StageOp::create(
        pipe_bldr_, loc_,
        /*depends_in=*/ValueRange{slots_.token(cfg.n_tokens - 1)},
        /*depends_out=*/ValueRange{});
    if (store_units_) {
      stage.setApplicableUnitsAttr(store_units_);
    }

    OpBuilder b(stage.getBody(), stage.getBody()->end());
    Value c0 = arith::ConstantIndexOp::create(b, loc_, 0);
    Value l1_idx = emitL1Idx(b, c0);

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

    for (unsigned i = 0; i < cfg.out_slot_types.size(); ++i) {
      Value fifo_out_slot = slots_.outSlot(i);
      auto fifo_type = cast<ktdf::FifoSlotType>(fifo_out_slot.getType());
      ktdf::DataTransferOp::create(
          b, loc_, fifo_out_slot, AffineMap{}, ValueRange{},
          ArrayRef<int64_t>{static_cast<int64_t>(fifo_type.getNumElements())},
          partial_memref, dest_id, dest_indices, dest_sizes);
    }
  }

 private:
  // -------------------------------------------------------------------------
  // Emit (batch_iv_ - 0) / 1 — the normalised affine form expected by the
  // backend for L1-slot index arithmetic.
  //
  // c0 must be emitted by the caller inside the stage body (ktdf.pipeline
  // only allows ktdf.stage and ktdf.private as immediate children, so
  // constants cannot be emitted at pipeline scope).
  // -------------------------------------------------------------------------
  Value emitL1Idx(OpBuilder& b, Value c0) {
    Value c1 = arith::ConstantIndexOp::create(b, loc_, 1);
    Value sub = arith::SubIOp::create(b, loc_, batch_iv_, c0);
    return arith::DivSIOp::create(b, loc_, sub, c1);
  }

  OpBuilder& pipe_bldr_;  // Builder positioned at the pipeline body end.
  Location loc_;          // Location for all emitted ops.
  Value batch_iv_;        // Outer batch scf.for induction variable.
  ArrayAttr load_units_;  // applicable_units for the Load stage (may be null).
  ArrayAttr
      compute_units_;  // applicable_units for the Compute stage (may be null).
  ArrayAttr
      store_units_;  // applicable_units for the Store stage (may be null).

  // Slot/token result wrapper bound via setSlots(); all stage builders address
  // slots and tokens through the FifoPrivateResult named accessors.
  FifoPrivateResult slots_;
};

// ---------------------------------------------------------------------------
// Pass implementation
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
  // Walk the module for linalg.generic ops that have at least one reduction
  // iterator.  For each such op, locate the immediately enclosing ktdf.stage
  // (the Compute stage) and collect unique stages into a list.  The list is
  // processed after the walk to avoid mutating IR while walking.
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

      // linalg.generic is always directly nested inside a ktdf.stage
      // (structural invariant enforced by StageCoarsening).
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
  // Entry point for one Compute stage.  Determines the reduction dims, chunk
  // counts, and per-dim chunk sizes, then delegates to rewriteComputeStage().
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

      // loop_num_chunks is only used for the debug log below; the actual
      // per-dim loops are driven by per_dim_num_chunks.
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
  // Replace inner_pipeline with nested scf.for loops — one per reduction
  // dimension whose num_chunks > 1.  Dimensions with num_chunks == 1 need no
  // loop; their IV is treated as the constant 0 for offset and condition
  // computation.
  //
  // For N dims with num_chunks[j] > 1 the emitted structure is:
  //
  //   scf.for %iv_0 = 0 to num_chunks[0] step 1 {
  //     scf.for %iv_1 = 0 to num_chunks[1] step 1 {
  //       ...
  //         %condition = (iv_0 == 0 && iv_1 == 0 && ...)
  //         ktdf.pipeline {
  //           ktdf.private { ... }   // FIFO slots + tokens
  //           ktdf.stage { ... }     // Load
  //           ktdf.stage { ... }     // Compute
  //           ktdf.stage { ... }     // Store
  //         }
  //     }
  //   }
  //
  // The original inner_pipeline is erased after the replacement is inserted.
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
    // Step 1: Trace dataflow to discover the source/destination memrefs and
    // the original FIFO slot types.
    //
    // Input path:
    //   input_memref → DataTransferOp → fifo_in → ReadFromFifoOp
    //               → generic.ins[0]
    //
    // Output path:
    //   generic.result(0) → WriteToFifoOp → fifo_out
    //               → DataTransferOp → partial_memref
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
    // Step 2: Emit the replacement structure.
    //
    // Insertion point is set just before inner_pipeline so the new loops land
    // in the correct position.  inner_pipeline is erased at the end (Step 3).
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
    Value batch_iv = batch_for.getInductionVar();

    // Materialise per-dim upper-bound constants before the outermost loop.
    // These are index-typed scf.for bounds and must live at the scf.for
    // scope, not inside any stage body.
    size_t n_dims = reduction_dims.size();
    SmallVector<Value> c_per_dim_num_chunks;
    for (int64_t nc : per_dim_num_chunks) {
      c_per_dim_num_chunks.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, nc));
    }

    // Build loops from outermost (dim 0) to innermost (dim n_dims-1).
    // dim_ivs[j] holds the loop IV when a loop was created for dim j, or
    // c0_loop when num_chunks[j]==1 (no loop for that dim).
    // After this block, inner_builder is positioned at the innermost loop
    // body (or at the original insertion point if no loops were generated).
    SmallVector<Value> dim_ivs(n_dims);
    OpBuilder inner_builder = rewriter;  // copy: same insertion point

    // c0_loop / c1_loop: loop-bound constants emitted at scf.for scope.
    // StageFactory emits its own c0/c1 inside each stage body where they
    // are legal (stage bodies are the only valid ktdf.pipeline children).
    Value c0_loop = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1_loop = arith::ConstantIndexOp::create(rewriter, loc, 1);

    for (size_t j = 0; j < n_dims; ++j) {
      if (per_dim_num_chunks[j] == 1) {
        // No loop for this dim; use constant 0 as the IV.
        dim_ivs[j] = c0_loop;
      } else {
        auto loop = scf::ForOp::create(inner_builder, loc, c0_loop,
                                       c_per_dim_num_chunks[j], c1_loop);
        dim_ivs[j] = loop.getInductionVar();
        // Descend into the loop body for the next (inner) level.
        inner_builder = OpBuilder(loop.getBody(), loop.getBody()->begin());
      }
    }

    // condition = AND of (iv_j == 0) for every dim whose loop was generated.
    // Dims with num_chunks==1 always have iv==c0 and are skipped to avoid
    // dead arith.cmpi constants.
    // Degenerate case (all dims have num_chunks==1): condition stays null;
    // emit arith.constant 1 (i1) rather than a trivially-true cmpi.
    Value condition;
    for (size_t j = 0; j < n_dims; ++j) {
      if (per_dim_num_chunks[j] == 1) continue;
      Value eq = arith::CmpIOp::create(
          inner_builder, loc, arith::CmpIPredicate::eq, dim_ivs[j], c0_loop);
      condition = condition
                      ? arith::AndIOp::create(inner_builder, loc, condition, eq)
                      : eq;
    }
    if (!condition) {
      condition = arith::ConstantIntOp::create(inner_builder, loc, /*value=*/1,
                                               /*width=*/1);
    }

    // One ktdf.pipeline per innermost chunk iteration.
    auto phase_pipeline = ktdf::PipelineOp::create(inner_builder, loc);
    OpBuilder pipe_bldr(phase_pipeline.getBody(),
                        phase_pipeline.getBody()->end());

    // partial_fifo_type carries output-tensor-sized elements in the
    // Load → Compute direction (same src/dest endpoints as fifo_in).
    auto partial_fifo_type = ktdf::FifoSlotType::get(
        context, in_fifo_type.getSrc(), in_fifo_type.getDest(),
        out_fifo_type.getNumElements(), out_fifo_type.getElementType());

    StageFactory factory(pipe_bldr, loc, batch_iv, load_units, compute_units,
                         store_units);

    // Describe the FIFO slot layout and token count for this chunk pipeline:
    //   in-slot  0  (chunk_in_fifo_type) : input chunk,  Load → Compute
    //   partial-slot 0 (partial_fifo_type): partial accum, Load → Compute
    //                                       (non-first iterations only)
    //   out-slot 0  (out_fifo_type)      : reduction result, Compute → Store
    //   token 0                          : Load signals Compute
    //   token 1                          : Compute signals Store
    ChunkPipelineConfig cfg;
    cfg.in_slot_types = {chunk_in_fifo_type};
    cfg.partial_slot_types = {partial_fifo_type};
    cfg.out_slot_types = {out_fifo_type};
    cfg.n_tokens = 2;

    factory.setSlots(factory.buildFifoPrivate(cfg));

    factory.buildLoadStage(condition, input_memref, input_memref_type,
                           partial_memref, output_memref_type, reduction_dims,
                           chunk_sizes, dim_ivs, cfg);

    factory.buildComputeStage(condition, chunk_input_tensor_type,
                              output_tensor_type, generic_op, cfg);

    factory.buildStoreStage(partial_memref, cfg);

    // ------------------------------------------------------------------
    // Step 3: Erase the original inner pipeline now that the replacement
    // has been inserted before it.
    // ------------------------------------------------------------------
    rewriter.eraseOp(inner_pipeline);

    return success();
  }
};

}  // namespace

auto mlir::ktdf::createReductionDimChunkingPass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionDimChunkingPass>();
}
