//===-- SpyreOpReductionConstraints.h ---------------------------*- c++ -*-===//
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
// Native PDL constraint functions for spyreop.slice_reduction dispatch.
//
// Each constraint inspects the `reduction_scope` attribute of the matched op
// and succeeds iff it matches the expected step.  The ordinal values correspond
// to the TableGen definition in SpyreOpEnums.td:
//
//   InSliceFmaSrc0Shuf = 0
//   InSliceFmaSrc2Shuf = 1
//   InSliceFmaBothShuf = 2
//   AcrossSliceSplat   = 3
//   AcrossSliceScanGap = 4
//   AcrossSliceFma16   = 5
//
// These functions are registered by the scheduler-local PatternCache subclass
// in ApplyDevicePatterns.cpp via registerNativeFunctions().
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_SPYREOPREDUCTIONCONSTRAINTS_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_SPYREOPREDUCTIONCONSTRAINTS_H_

// PDL types (PDLValue, PDLResultList, PatternRewriter) are provided by
// ApplyPatterns.h which every caller of these functions already includes.
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/ApplyPatterns.h"
#include "ktir/Dialect/SpyreOp/SpyreOpAttrs.h"

namespace scheduler {

// Returns the integer ordinal of a spyreop.reduction_scope attribute, or
// std::nullopt if the attribute is not of that type.
inline auto getReductionScopeOrdinal(mlir::Attribute attr)
    -> std::optional<int64_t> {
  if (!attr) return std::nullopt;
  if (attr.getAbstractAttribute().getName() != "spyreop.reduction_scope")
    return std::nullopt;
  if (auto scope = mlir::dyn_cast<mlir::spyreop::ReductionScopeAttr>(attr))
    return static_cast<int64_t>(scope.getValue());
  return std::nullopt;
}

// Group constraints — succeed for any step belonging to one phase.

inline auto spyreIsInSlice(mlir::PatternRewriter& /*rewriter*/,
                           mlir::PDLResultList& /*results*/,
                           llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 1);
  auto ordinal = getReductionScopeOrdinal(values[0].cast<mlir::Attribute>());
  return mlir::success(ordinal && *ordinal >= 0 && *ordinal <= 2);
}

inline auto spyreIsAcrossSlice(mlir::PatternRewriter& /*rewriter*/,
                               mlir::PDLResultList& /*results*/,
                               llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 1);
  auto ordinal = getReductionScopeOrdinal(values[0].cast<mlir::Attribute>());
  return mlir::success(ordinal && *ordinal >= 3 && *ordinal <= 5);
}

// Per-step constraints — one per enum case.

inline auto spyreIsInSliceFmaSrc0Shuf(mlir::PatternRewriter& /*rewriter*/,
                                      mlir::PDLResultList& /*results*/,
                                      llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 1);
  auto ordinal = getReductionScopeOrdinal(values[0].cast<mlir::Attribute>());
  return mlir::success(ordinal && *ordinal == 0);
}

inline auto spyreIsInSliceFmaSrc2Shuf(mlir::PatternRewriter& /*rewriter*/,
                                      mlir::PDLResultList& /*results*/,
                                      llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 1);
  auto ordinal = getReductionScopeOrdinal(values[0].cast<mlir::Attribute>());
  return mlir::success(ordinal && *ordinal == 1);
}

inline auto spyreIsInSliceFmaBothShuf(mlir::PatternRewriter& /*rewriter*/,
                                      mlir::PDLResultList& /*results*/,
                                      llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 1);
  auto ordinal = getReductionScopeOrdinal(values[0].cast<mlir::Attribute>());
  return mlir::success(ordinal && *ordinal == 2);
}

inline auto spyreIsAcrossSliceSplat(mlir::PatternRewriter& /*rewriter*/,
                                    mlir::PDLResultList& /*results*/,
                                    llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 1);
  auto ordinal = getReductionScopeOrdinal(values[0].cast<mlir::Attribute>());
  return mlir::success(ordinal && *ordinal == 3);
}

inline auto spyreIsAcrossSliceScanGap(mlir::PatternRewriter& /*rewriter*/,
                                      mlir::PDLResultList& /*results*/,
                                      llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 1);
  auto ordinal = getReductionScopeOrdinal(values[0].cast<mlir::Attribute>());
  return mlir::success(ordinal && *ordinal == 4);
}

inline auto spyreIsAcrossSliceFma16(mlir::PatternRewriter& /*rewriter*/,
                                    mlir::PDLResultList& /*results*/,
                                    llvm::ArrayRef<mlir::PDLValue> values)
    -> mlir::LogicalResult {
  assert(values.size() == 1);
  auto ordinal = getReductionScopeOrdinal(values[0].cast<mlir::Attribute>());
  return mlir::success(ordinal && *ordinal == 5);
}

// Registers all spyreop.slice_reduction PDL constraints into @p patterns .
// Call this from PatternCache::registerNativeFunctions.
inline void registerSpyreOpReductionConstraints(
    mlir::PDLPatternModule& patterns) {
  // Group-level constraints (match all steps of one direction).
  patterns.registerConstraintFunction("spyreop.is_in_slice", spyreIsInSlice);
  patterns.registerConstraintFunction("spyreop.is_across_slice",
                                      spyreIsAcrossSlice);
  // Per-step constraints.
  patterns.registerConstraintFunction("spyreop.is_in_slice_fma_src0_shuf",
                                      spyreIsInSliceFmaSrc0Shuf);
  patterns.registerConstraintFunction("spyreop.is_in_slice_fma_src2_shuf",
                                      spyreIsInSliceFmaSrc2Shuf);
  patterns.registerConstraintFunction("spyreop.is_in_slice_fma_both_shuf",
                                      spyreIsInSliceFmaBothShuf);
  patterns.registerConstraintFunction("spyreop.is_across_slice_splat",
                                      spyreIsAcrossSliceSplat);
  patterns.registerConstraintFunction("spyreop.is_across_slice_scan_gap",
                                      spyreIsAcrossSliceScanGap);
  patterns.registerConstraintFunction("spyreop.is_across_slice_fma16",
                                      spyreIsAcrossSliceFma16);
}

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_SPYREOPREDUCTIONCONSTRAINTS_H_
