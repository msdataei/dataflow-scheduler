// RUN: dataflow-scheduler-opt --reduction-dim-chunking="chunk-size-threshold=8192"  %s | FileCheck %s --check-prefix=CHECK-N4
// RUN: dataflow-scheduler-opt --reduction-dim-chunking="chunk-size-threshold=16384" %s | FileCheck %s --check-prefix=CHECK-N2

// Test: auto-infer num_chunks from chunk-size-threshold (no --num-chunks flag).
//
// Input: tensor<1x256x64xf16>  (reduction dim 1, size 256)
//   total_input_bytes = 1 * 256 * 64 * 2 = 32768
//
//   threshold=8192  → smallest N with 32768/N ≤ 8192  is N=4 → chunk_size=64
//   threshold=16384 → smallest N with 32768/N ≤ 16384 is N=2 → chunk_size=128
//
// The two RUN lines exercise the same input and verify that distinct thresholds
// produce distinct loop bounds and FIFO sizes, confirming ReductionChunkAnalysis
// drives the pass correctly in auto-infer mode.

// ---------------------------------------------------------------------------
// threshold=8192 → N=4 chunks, fifo_in=4096xf16, chunk loop bound=4
// ---------------------------------------------------------------------------

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK-N4: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-N4: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-N4: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-N4: #[[$ATTR_3:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-N4-LABEL:   module {
// CHECK-N4:           func.func @sum_1core() attributes {grid = [1]} {
// CHECK-N4:             call @local_schedule_0() : () -> ()
// CHECK-N4:             return
// CHECK-N4:           }
// CHECK-N4:           func.func private @local_schedule_0()
// CHECK-N4:         }
// CHECK-N4:         ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-N4-LABEL:   module @local_schedule_0 {
// CHECK-N4:           func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-N4:             %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-N4:             %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-N4:             %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-N4:             %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK-N4:             %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
// CHECK-N4:             %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
// CHECK-N4:             %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-N4:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK-N4:             %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-N4:             %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2x64xf16> to memref<2x64xf16, "DDR">
// CHECK-N4:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
// CHECK-N4:             %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-N4:             ktdf.pipeline {
// CHECK-N4:               %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-N4:                 %[[ALLOC_0:.*]] = memref.alloc() : memref<2x1x256x64xf16, "L1">
// CHECK-N4:                 %[[ALLOC_1:.*]] = memref.alloc() : memref<2x1x64xf16, "L1">
// CHECK-N4:                 %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-N4:                 %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-N4:                 ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-N4:               }
// CHECK-N4:               ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-N4:                 scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-N4:                   %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-N4:                   %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK-N4:                   ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[VAL_1]], 0, %[[CONSTANT_0]] * 64] size [1, 256, 64] to %[[VAL_0]]#0{{\[}}%[[DIVSI_0]], 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x1x256x64xf16, "L1">
// CHECK-N4:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-N4:               } {applicable_units = ["MNILU"]}
// CHECK-N4:               ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-N4:                 %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-N4:                 %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-N4:                 %[[CONSTANT_6:.*]] = arith.constant 64 : index
// CHECK-N4:                 %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK-N4:                 %[[CONSTANT_8:.*]] = arith.constant 4 : index
// CHECK-N4:                 %[[CONSTANT_9:.*]] = arith.constant 4 : index
// CHECK-N4:                 %[[CONSTANT_10:.*]] = arith.constant 3 : index
// CHECK-N4:                 scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-N4:                   %[[CONSTANT_11:.*]] = arith.constant 0 : index
// CHECK-N4:                   %[[CONSTANT_12:.*]] = arith.constant 1 : index
// CHECK-N4:                   %[[CONSTANT_13:.*]] = arith.constant 4 : index
// CHECK-N4:                   %[[CONSTANT_14:.*]] = arith.constant 64 : index
// CHECK-N4:                   %[[CONSTANT_15:.*]] = arith.constant 1 : index
// CHECK-N4:                   %[[CONSTANT_16:.*]] = arith.constant 4 : index
// CHECK-N4:                   %[[CONSTANT_17:.*]] = arith.constant 3 : index
// CHECK-N4:                   scf.for %[[VAL_4:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_13]] step %[[CONSTANT_12]] {
// CHECK-N4:                     %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_4]], %[[CONSTANT_11]] : index
// CHECK-N4:                     ktdf.pipeline {
// CHECK-N4:                       %[[PRIVATE_1:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-N4:                         %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
// CHECK-N4:                         %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-N4:                         %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-N4:                         %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-N4:                         ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-N4:                       }
// CHECK-N4:                       ktdf.stage depends_in(none) depends_out(%[[VAL_5:.*]]#2) {
// CHECK-N4:                         %[[SUBI_1:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_4]] : index
// CHECK-N4:                         %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_5]] : index
// CHECK-N4:                         %[[DIVSI_2:.*]] = arith.divsi %[[VAL_4]], %[[CONSTANT_15]] : index
// CHECK-N4:                         %[[REMSI_0:.*]] = arith.remsi %[[DIVSI_2]], %[[CONSTANT_16]] : index
// CHECK-N4:                         %[[MULI_0:.*]] = arith.muli %[[REMSI_0]], %[[CONSTANT_14]] : index
// CHECK-N4:                         ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[DIVSI_1]], %[[CONSTANT_11]], %[[MULI_0]], %[[CONSTANT_11]]] size [1, 1, 64, 64] to %[[VAL_5]]#0 size [4096] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
// CHECK-N4:                         scf.if %[[CMPI_0]] {
// CHECK-N4:                         } else {
// CHECK-N4:                           ktdf.data_transfer from %[[VAL_2]]#1{{\[}}%[[DIVSI_1]], %[[CONSTANT_11]], %[[CONSTANT_11]]] size [1, 1, 64] to %[[VAL_5]]#1 size [64] : memref<2x1x64xf16, "L1">, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-N4:                         }
// CHECK-N4:                       } {applicable_units = ["L1LU"]}
// CHECK-N4:                       ktdf.stage depends_in(%[[VAL_6:.*]]#2) depends_out(%[[VAL_6]]#3) {
// CHECK-N4:                         %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_6]]#0 : <"L1LU" -> "SFU", 4096xf16> -> tensor<1x64x64xf16>
// CHECK-N4:                         %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (tensor<1x64xf16>) {
// CHECK-N4:                           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-N4:                           scf.yield %[[EMPTY_0]] : tensor<1x64xf16>
// CHECK-N4:                         } else {
// CHECK-N4:                           %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[VAL_6]]#1 : <"SFU" -> "L1SU", 64xf16> -> tensor<1x64xf16>
// CHECK-N4:                           scf.yield %[[READ_FROM_FIFO_1]] : tensor<1x64xf16>
// CHECK-N4:                         }
// CHECK-N4:                         %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x64x64xf16>) outs(%[[IF_0]] : tensor<1x64xf16>) {
// CHECK-N4:                         ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16):
// CHECK-N4:                           %[[ADDF_0:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f16
// CHECK-N4:                           linalg.yield %[[ADDF_0]] : f16
// CHECK-N4:                         } -> tensor<1x64xf16>
// CHECK-N4:                         ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_6]]#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-N4:                       } {applicable_units = ["SFU"]}
// CHECK-N4:                       ktdf.stage depends_in(%[[VAL_9:.*]]#3) depends_out(none) {
// CHECK-N4:                         %[[SUBI_2:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_4]] : index
// CHECK-N4:                         %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_5]] : index
// CHECK-N4:                         ktdf.data_transfer from %[[VAL_9]]#1 size [64] to %[[VAL_2]]#1{{\[}}%[[DIVSI_3]], %[[CONSTANT_11]], %[[CONSTANT_11]]] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK-N4:                       } {applicable_units = ["L1SU"]}
// CHECK-N4:                     }
// CHECK-N4:                   }
// CHECK-N4:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-N4:               } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-N4:               ktdf.stage depends_in(%[[VAL_10:.*]]#3) depends_out(none) {
// CHECK-N4:                 scf.for %[[VAL_11:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-N4:                   %[[SUBI_3:.*]] = arith.subi %[[VAL_11]], %[[CONSTANT_0]] : index
// CHECK-N4:                   %[[DIVSI_4:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : index
// CHECK-N4:                   ktdf.data_transfer from %[[VAL_10]]#1{{\[}}%[[DIVSI_4]], 0, 0] size [1, 1, 64] to %[[CAST_1]]{{\[}}%[[VAL_11]], %[[CONSTANT_0]] * 64] size [1, 64] : memref<2x1x64xf16, "L1">, memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-N4:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-N4:               } {applicable_units = ["MNISU"]}
// CHECK-N4:             }
// CHECK-N4:             return
// CHECK-N4:           }
// CHECK-N4:         }


// ---------------------------------------------------------------------------
// threshold=16384 → N=2 chunks, fifo_in=8192xf16, chunk loop bound=2
// ---------------------------------------------------------------------------

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK-N2: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-N2: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-N2: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-N2: #[[$ATTR_3:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-N2-LABEL:   module {
// CHECK-N2:           func.func @sum_1core() attributes {grid = [1]} {
// CHECK-N2:             call @local_schedule_0() : () -> ()
// CHECK-N2:             return
// CHECK-N2:           }
// CHECK-N2:           func.func private @local_schedule_0()
// CHECK-N2:         }
// CHECK-N2:         ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-N2-LABEL:   module @local_schedule_0 {
// CHECK-N2:           func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-N2:             %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-N2:             %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-N2:             %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-N2:             %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK-N2:             %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
// CHECK-N2:             %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
// CHECK-N2:             %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-N2:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK-N2:             %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-N2:             %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2x64xf16> to memref<2x64xf16, "DDR">
// CHECK-N2:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
// CHECK-N2:             %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-N2:             ktdf.pipeline {
// CHECK-N2:               %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-N2:                 %[[ALLOC_0:.*]] = memref.alloc() : memref<2x1x256x64xf16, "L1">
// CHECK-N2:                 %[[ALLOC_1:.*]] = memref.alloc() : memref<2x1x64xf16, "L1">
// CHECK-N2:                 %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-N2:                 %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-N2:                 ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-N2:               }
// CHECK-N2:               ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-N2:                 scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-N2:                   %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-N2:                   %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK-N2:                   ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[VAL_1]], 0, %[[CONSTANT_0]] * 64] size [1, 256, 64] to %[[VAL_0]]#0{{\[}}%[[DIVSI_0]], 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x1x256x64xf16, "L1">
// CHECK-N2:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-N2:               } {applicable_units = ["MNILU"]}
// CHECK-N2:               ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-N2:                 %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-N2:                 %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-N2:                 %[[CONSTANT_6:.*]] = arith.constant 128 : index
// CHECK-N2:                 %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK-N2:                 %[[CONSTANT_8:.*]] = arith.constant 2 : index
// CHECK-N2:                 %[[CONSTANT_9:.*]] = arith.constant 2 : index
// CHECK-N2:                 %[[CONSTANT_10:.*]] = arith.constant 1 : index
// CHECK-N2:                 scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-N2:                   %[[CONSTANT_11:.*]] = arith.constant 0 : index
// CHECK-N2:                   %[[CONSTANT_12:.*]] = arith.constant 1 : index
// CHECK-N2:                   %[[CONSTANT_13:.*]] = arith.constant 2 : index
// CHECK-N2:                   %[[CONSTANT_14:.*]] = arith.constant 128 : index
// CHECK-N2:                   %[[CONSTANT_15:.*]] = arith.constant 1 : index
// CHECK-N2:                   %[[CONSTANT_16:.*]] = arith.constant 2 : index
// CHECK-N2:                   %[[CONSTANT_17:.*]] = arith.constant 1 : index
// CHECK-N2:                   scf.for %[[VAL_4:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_13]] step %[[CONSTANT_12]] {
// CHECK-N2:                     %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_4]], %[[CONSTANT_11]] : index
// CHECK-N2:                     ktdf.pipeline {
// CHECK-N2:                       %[[PRIVATE_1:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-N2:                         %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>
// CHECK-N2:                         %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-N2:                         %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-N2:                         %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-N2:                         ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-N2:                       }
// CHECK-N2:                       ktdf.stage depends_in(none) depends_out(%[[VAL_5:.*]]#2) {
// CHECK-N2:                         %[[SUBI_1:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_4]] : index
// CHECK-N2:                         %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_5]] : index
// CHECK-N2:                         %[[DIVSI_2:.*]] = arith.divsi %[[VAL_4]], %[[CONSTANT_15]] : index
// CHECK-N2:                         %[[REMSI_0:.*]] = arith.remsi %[[DIVSI_2]], %[[CONSTANT_16]] : index
// CHECK-N2:                         %[[MULI_0:.*]] = arith.muli %[[REMSI_0]], %[[CONSTANT_14]] : index
// CHECK-N2:                         ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[DIVSI_1]], %[[CONSTANT_11]], %[[MULI_0]], %[[CONSTANT_11]]] size [1, 1, 128, 64] to %[[VAL_5]]#0 size [8192] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>
// CHECK-N2:                         scf.if %[[CMPI_0]] {
// CHECK-N2:                         } else {
// CHECK-N2:                           ktdf.data_transfer from %[[VAL_2]]#1{{\[}}%[[DIVSI_1]], %[[CONSTANT_11]], %[[CONSTANT_11]]] size [1, 1, 64] to %[[VAL_5]]#1 size [64] : memref<2x1x64xf16, "L1">, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-N2:                         }
// CHECK-N2:                       } {applicable_units = ["L1LU"]}
// CHECK-N2:                       ktdf.stage depends_in(%[[VAL_6:.*]]#2) depends_out(%[[VAL_6]]#3) {
// CHECK-N2:                         %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_6]]#0 : <"L1LU" -> "SFU", 8192xf16> -> tensor<1x128x64xf16>
// CHECK-N2:                         %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (tensor<1x64xf16>) {
// CHECK-N2:                           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-N2:                           scf.yield %[[EMPTY_0]] : tensor<1x64xf16>
// CHECK-N2:                         } else {
// CHECK-N2:                           %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[VAL_6]]#1 : <"SFU" -> "L1SU", 64xf16> -> tensor<1x64xf16>
// CHECK-N2:                           scf.yield %[[READ_FROM_FIFO_1]] : tensor<1x64xf16>
// CHECK-N2:                         }
// CHECK-N2:                         %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x128x64xf16>) outs(%[[IF_0]] : tensor<1x64xf16>) {
// CHECK-N2:                         ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16):
// CHECK-N2:                           %[[ADDF_0:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f16
// CHECK-N2:                           linalg.yield %[[ADDF_0]] : f16
// CHECK-N2:                         } -> tensor<1x64xf16>
// CHECK-N2:                         ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_6]]#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-N2:                       } {applicable_units = ["SFU"]}
// CHECK-N2:                       ktdf.stage depends_in(%[[VAL_9:.*]]#3) depends_out(none) {
// CHECK-N2:                         %[[SUBI_2:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_4]] : index
// CHECK-N2:                         %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_5]] : index
// CHECK-N2:                         ktdf.data_transfer from %[[VAL_9]]#1 size [64] to %[[VAL_2]]#1{{\[}}%[[DIVSI_3]], %[[CONSTANT_11]], %[[CONSTANT_11]]] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK-N2:                       } {applicable_units = ["L1SU"]}
// CHECK-N2:                     }
// CHECK-N2:                   }
// CHECK-N2:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-N2:               } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-N2:               ktdf.stage depends_in(%[[VAL_10:.*]]#3) depends_out(none) {
// CHECK-N2:                 scf.for %[[VAL_11:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-N2:                   %[[SUBI_3:.*]] = arith.subi %[[VAL_11]], %[[CONSTANT_0]] : index
// CHECK-N2:                   %[[DIVSI_4:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : index
// CHECK-N2:                   ktdf.data_transfer from %[[VAL_10]]#1{{\[}}%[[DIVSI_4]], 0, 0] size [1, 1, 64] to %[[CAST_1]]{{\[}}%[[VAL_11]], %[[CONSTANT_0]] * 64] size [1, 64] : memref<2x1x64xf16, "L1">, memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-N2:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-N2:               } {applicable_units = ["MNISU"]}
// CHECK-N2:             }
// CHECK-N2:             return
// CHECK-N2:           }
// CHECK-N2:         }


#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  module {
    func.func @sum_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c2 = arith.constant 2 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<2x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2x64xf16> to memref<2x64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<2x1x256x64xf16, "L1">
          %alloc_3 = memref.alloc() : memref<2x1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %cast[%arg0, 0, %c0 * 64] size [1, 256, 64] to %2#0[%4, 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x1x256x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %2#0[%5, 0, 0, 0] size [1, 1, 256, 64] to %3#0 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
                %5 = tensor.empty() : tensor<1x64xf16>
                %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%4 : tensor<1x256x64xf16>) outs(%5 : tensor<1x64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %7 = arith.addf %in, %out : f16
                  linalg.yield %7 : f16
                } -> tensor<1x64xf16>
                ktdf.write_to_fifo %6, %3#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %3#1 size [64] to %2#1[%5, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %2#1[%4, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<2x1x64xf16, "L1">, memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
