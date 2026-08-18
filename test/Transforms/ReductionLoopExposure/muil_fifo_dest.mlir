// RUN: dataflow-scheduler-opt --reduction-loop-exposure %s | FileCheck %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   module {
// CHECK:     func.func @sum_1core() attributes {grid = [1]} {
// CHECK:       call @local_schedule_0() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @local_schedule_0()
// CHECK:   }
// CHECK:   ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2x64xf16> to memref<2x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x1x256x64xf16, "L1">
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<2x1x64xf16, "L1">
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[VAL_1]], 0, %[[CONSTANT_0]] * 64] size [1, 256, 64] to %[[VAL_0]]#0{{\[}}%[[DIVSI_0]], 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x1x256x64xf16, "L1">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-NEXT:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:           %[[CONSTANT_6:.*]] = arith.constant 64 : index
// CHECK-NEXT:           %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK-NEXT:           %[[CONSTANT_8:.*]] = arith.constant 4 : index
// CHECK-NEXT:           %[[CONSTANT_9:.*]] = arith.constant 4 : index
// CHECK-NEXT:           %[[CONSTANT_10:.*]] = arith.constant 3 : index
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[CONSTANT_11:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[CONSTANT_12:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[CONSTANT_13:.*]] = arith.constant 4 : index
// CHECK-NEXT:             %[[CONSTANT_14:.*]] = arith.constant 64 : index
// CHECK-NEXT:             %[[CONSTANT_15:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[CONSTANT_16:.*]] = arith.constant 4 : index
// CHECK-NEXT:             %[[CONSTANT_17:.*]] = arith.constant 3 : index
// CHECK-NEXT:             scf.for %[[VAL_4:.*]] = %[[CONSTANT_11]] to %[[CONSTANT_13]] step %[[CONSTANT_12]] {
// CHECK-NEXT:               %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_4]], %[[CONSTANT_11]] : index
// CHECK-NEXT:               %[[CONSTANT_18:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[CONSTANT_19:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[CONSTANT_20:.*]] = arith.constant 63 : index
// CHECK-NEXT:               %[[CONSTANT_21:.*]] = arith.constant 0 : index
// CHECK-NEXT:               %[[CONSTANT_22:.*]] = arith.constant 1 : index
// CHECK-NEXT:               %[[CONSTANT_23:.*]] = arith.constant 0 : index
// CHECK-NEXT:               ktdf.pipeline {
// CHECK-NEXT:                 %[[PRIVATE_1:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                   %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                   %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                   %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                   %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                   ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:                 }
// CHECK-NEXT:                 ktdf.stage depends_in(none) depends_out(%[[VAL_5:.*]]#2) {
// CHECK-NEXT:                   %[[CONSTANT_24:.*]] = arith.constant 1 : index
// CHECK-NEXT:                   scf.for %[[VAL_6:.*]] = %[[CONSTANT_21]] to %[[CONSTANT_24]] step %[[CONSTANT_22]] {
// CHECK-NEXT:                     %[[CONSTANT_25:.*]] = arith.constant 64 : index
// CHECK-NEXT:                     scf.for %[[VAL_7:.*]] = %[[CONSTANT_18]] to %[[CONSTANT_25]] step %[[CONSTANT_19]] {
// CHECK-NEXT:                       %[[SUBI_1:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_4]] : index
// CHECK-NEXT:                       %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_5]] : index
// CHECK-NEXT:                       %[[DIVSI_2:.*]] = arith.divsi %[[VAL_4]], %[[CONSTANT_15]] : index
// CHECK-NEXT:                       %[[REMSI_0:.*]] = arith.remsi %[[DIVSI_2]], %[[CONSTANT_16]] : index
// CHECK-NEXT:                       %[[MULI_0:.*]] = arith.muli %[[REMSI_0]], %[[CONSTANT_14]] : index
// CHECK-NEXT:                       ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[DIVSI_1]], %[[CONSTANT_11]], %[[VAL_7]] + %[[VAL_6]], %[[CONSTANT_11]]] size [1, 1, 1, 64] to %[[VAL_5]]#0 size [64] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                       scf.if %[[CMPI_0]] {
// CHECK-NEXT:                       } else {
// CHECK-NEXT:                         ktdf.data_transfer from %[[VAL_2]]#1{{\[}}%[[DIVSI_1]], %[[VAL_7]] + %[[VAL_6]], %[[CONSTANT_11]]] size [1, 1, 64] to %[[VAL_5]]#1 size [64] : memref<2x1x64xf16, "L1">, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:                 } {applicable_units = ["L1LU"]}
// CHECK-NEXT:                 ktdf.stage depends_in(%[[VAL_8:.*]]#2) depends_out(%[[VAL_8]]#3) {
// CHECK-NEXT:                   %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:                   %[[CONSTANT_26:.*]] = arith.constant 1 : index
// CHECK-NEXT:                   %[[FOR_0:.*]] = scf.for %[[VAL_9:.*]] = %[[CONSTANT_21]] to %[[CONSTANT_26]] step %[[CONSTANT_22]] iter_args(%[[VAL_10:.*]] = %[[EMPTY_0]]) -> (tensor<1x64xf16>) {
// CHECK-NEXT:                     %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_8]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
// CHECK-NEXT:                     %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x1x64xf16>) outs(%[[VAL_10]] : tensor<1x64xf16>) {
// CHECK-NEXT:                     ^bb0(%[[VAL_11:.*]]: f16, %[[VAL_12:.*]]: f16):
// CHECK-NEXT:                       %[[ADDF_0:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] : f16
// CHECK-NEXT:                       linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                     } -> tensor<1x64xf16>
// CHECK-NEXT:                     %[[CMPI_1:.*]] = arith.cmpi eq, %[[VAL_9]], %[[CONSTANT_23]] : index
// CHECK-NEXT:                     scf.if %[[CMPI_1]] {
// CHECK-NEXT:                       ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_8]]#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                     }
// CHECK-NEXT:                     scf.yield %[[GENERIC_0]] : tensor<1x64xf16>
// CHECK-NEXT:                   } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:                 } {applicable_units = ["SFU"]}
// CHECK-NEXT:                 ktdf.stage depends_in(%[[VAL_13:.*]]#3) depends_out(none) {
// CHECK-NEXT:                   %[[CONSTANT_27:.*]] = arith.constant 1 : index
// CHECK-NEXT:                   scf.for %[[VAL_14:.*]] = %[[CONSTANT_21]] to %[[CONSTANT_27]] step %[[CONSTANT_22]] {
// CHECK-NEXT:                     %[[CONSTANT_28:.*]] = arith.constant 64 : index
// CHECK-NEXT:                     scf.for %[[VAL_15:.*]] = %[[CONSTANT_18]] to %[[CONSTANT_28]] step %[[CONSTANT_19]] {
// CHECK-NEXT:                       %[[SUBI_2:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_4]] : index
// CHECK-NEXT:                       %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_5]] : index
// CHECK-NEXT:                       %[[CMPI_2:.*]] = arith.cmpi eq, %[[VAL_15]], %[[CONSTANT_20]] : index
// CHECK-NEXT:                       scf.if %[[CMPI_2]] {
// CHECK-NEXT:                         %[[CMPI_3:.*]] = arith.cmpi eq, %[[VAL_14]], %[[CONSTANT_23]] : index
// CHECK-NEXT:                         scf.if %[[CMPI_3]] {
// CHECK-NEXT:                           ktdf.data_transfer from %[[VAL_13]]#1 size [64] to %[[VAL_2]]#1{{\[}}%[[DIVSI_3]], %[[CONSTANT_11]], %[[CONSTANT_11]]] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:                 } {applicable_units = ["L1SU"]}
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_16:.*]]#3) depends_out(none) {
// CHECK-NEXT:           scf.for %[[VAL_17:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_3:.*]] = arith.subi %[[VAL_17]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_4:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_16]]#1{{\[}}%[[DIVSI_4]], 0, 0] size [1, 1, 64] to %[[CAST_1]]{{\[}}%[[VAL_17]], %[[CONSTANT_0]] * 64] size [1, 64] : memref<2x1x64xf16, "L1">, memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }





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
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
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
          %c0_3 = arith.constant 0 : index
          %c1_4 = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %c1_5 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c4_6 = arith.constant 4 : index
          %c3 = arith.constant 3 : index
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %c0_7 = arith.constant 0 : index
            %c1_8 = arith.constant 1 : index
            %c4_9 = arith.constant 4 : index
            %c64_10 = arith.constant 64 : index
            %c1_11 = arith.constant 1 : index
            %c4_12 = arith.constant 4 : index
            %c3_13 = arith.constant 3 : index
            scf.for %arg1 = %c0_7 to %c4_9 step %c1_8 {
              %3 = arith.cmpi eq, %arg1, %c0_7 : index
              %c0_14 = arith.constant 0 : index
              %c1_15 = arith.constant 1 : index
              %c63 = arith.constant 63 : index
              ktdf.pipeline {
                %4:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                  %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  %6 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                  %7 = ktdf.create_token : !ktdf.token
                  %8 = ktdf.create_token : !ktdf.token
                  ktdf.private_yield %5, %6, %7, %8 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
                }
                ktdf.stage depends_in(none) depends_out(%4#2) {
                  %c64_16 = arith.constant 64 : index
                  scf.for %arg2 = %c0_14 to %c64_16 step %c1_15 {
                    %5 = arith.subi %arg0, %c0_3 : index
                    %6 = arith.divsi %5, %c1_4 : index
                    %7 = arith.divsi %arg1, %c1_11 : index
                    %8 = arith.remsi %7, %c4_12 : index
                    %9 = arith.muli %8, %c64_10 : index
                    ktdf.data_transfer from %2#0[%6, %c0_7, %arg2, %c0_7] size [1, 1, 1, 64] to %4#0 size [64] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    scf.if %3 {
                    } else {
                      ktdf.data_transfer from %2#1[%6, %arg2, %c0_7] size [1, 1, 64] to %4#1 size [64] : memref<2x1x64xf16, "L1">, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                    }
                  }
                } {applicable_units = ["L1LU"]}
                ktdf.stage depends_in(%4#2) depends_out(%4#3) {
                  %5 = tensor.empty() : tensor<1x64xf16>
                  %c64_16 = arith.constant 64 : index
                  %6 = scf.for %arg2 = %c0_14 to %c64_16 step %c1_15 iter_args(%arg3 = %5) -> (tensor<1x64xf16>) {
                    %7 = ktdf.read_from_fifo %4#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%7 : tensor<1x1x64xf16>) outs(%arg3 : tensor<1x64xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %10 = arith.addf %in, %out : f16
                      linalg.yield %10 : f16
                    } -> tensor<1x64xf16>
                    %9 = arith.cmpi eq, %arg2, %c63 : index
                    scf.if %9 {
                      ktdf.write_to_fifo %8, %4#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                    }
                    scf.yield %8 : tensor<1x64xf16>
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                } {applicable_units = ["SFU"]}
                ktdf.stage depends_in(%4#3) depends_out(none) {
                  %c64_16 = arith.constant 64 : index
                  scf.for %arg2 = %c0_14 to %c64_16 step %c1_15 {
                    %5 = arith.subi %arg0, %c0_3 : index
                    %6 = arith.divsi %5, %c1_4 : index
                    %7 = arith.cmpi eq, %arg2, %c63 : index
                    scf.if %7 {
                      ktdf.data_transfer from %4#1 size [64] to %2#1[%6, %c0_7, %c0_7] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
                    }
                  }
                } {applicable_units = ["L1SU"]}
              }
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

