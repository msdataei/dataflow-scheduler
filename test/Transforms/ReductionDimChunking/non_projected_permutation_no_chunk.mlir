// RUN: dataflow-scheduler-opt --reduction-dim-chunking %s | FileCheck %s

// A reduction whose input is not indexed by a projected permutation is
// rejected only when it has to be chunked.  This windowed sum (input dim 0 is
// indexed by d0 + d1, as in a convolution) is well under the chunk size
// threshold, so every reduction dim gets one chunk and the pass leaves the
// pipeline as it is instead of failing.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @windowed_sum(
// CHECK-SAME:      %[[ARG0:.*]]: memref<7x4x64xf16, "L1">,
// CHECK-SAME:      %[[ARG1:.*]]: memref<4x64xf16, "L1">) {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:         %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>
// CHECK-NEXT:         %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:         ktdf.data_transfer from %[[ARG0]][0, 0, 0] size [7, 4, 64] to %[[VAL_0]]#0 size [1792] : memref<7x4x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>
// CHECK-NEXT:       } {applicable_units = ["L1LU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_1:.*]]#2) depends_out(%[[VAL_1]]#3) {
// CHECK-NEXT:         %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_1]]#0 : <"L1LU" -> "SFU", 1792xf16> -> tensor<7x4x64xf16>
// CHECK-NEXT:         %[[EMPTY_0:.*]] = tensor.empty() : tensor<4x64xf16>
// CHECK-NEXT:         %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<7x4x64xf16>) outs(%[[EMPTY_0]] : tensor<4x64xf16>) {
// CHECK-NEXT:         ^bb0(%[[VAL_2:.*]]: f16, %[[VAL_3:.*]]: f16):
// CHECK-NEXT:           %[[ADDF_0:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : f16
// CHECK-NEXT:           linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:         } -> tensor<4x64xf16>
// CHECK-NEXT:         ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_1]]#1 : tensor<4x64xf16>, <"SFU" -> "L1SU", 256xf16>
// CHECK-NEXT:       } {applicable_units = ["SFU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_4:.*]]#3) depends_out(none) {
// CHECK-NEXT:         ktdf.data_transfer from %[[VAL_4]]#1 size [256] to %[[ARG1]][0, 0] size [4, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, memref<4x64xf16, "L1">
// CHECK-NEXT:       } {applicable_units = ["L1SU"]}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// A reduction whose input is not indexed by a projected permutation is
// rejected only when it has to be chunked.  This windowed sum (input dim 0 is
// indexed by d0 + d1, as in a convolution) is well under the chunk size
// threshold, so every reduction dim gets one chunk and the pass leaves the
// pipeline as it is instead of failing.

#in  = affine_map<(d0, d1, d2) -> (d0 + d1, d1, d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @windowed_sum(%src: memref<7x4x64xf16, "L1">, %dst: memref<4x64xf16, "L1">) {
    ktdf.pipeline {
      %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.token, !ktdf.token) {
        %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>
        %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %i, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#2) {
        ktdf.data_transfer from %src[0, 0, 0] size [7, 4, 64] to %p#0 size [1792] : memref<7x4x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%p#2) depends_out(%p#3) {
        %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 1792xf16> -> tensor<7x4x64xf16>
        %e = tensor.empty() : tensor<4x64xf16>
        %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "parallel"]} ins(%x : tensor<7x4x64xf16>) outs(%e : tensor<4x64xf16>) {
        ^bb0(%a: f16, %acc: f16):
          %s = arith.addf %a, %acc : f16
          linalg.yield %s : f16
        } -> tensor<4x64xf16>
        ktdf.write_to_fifo %r, %p#1 : tensor<4x64xf16>, <"SFU" -> "L1SU", 256xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#3) depends_out(none) {
        ktdf.data_transfer from %p#1 size [256] to %dst[0, 0] size [4, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, memref<4x64xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
    return
  }
}
