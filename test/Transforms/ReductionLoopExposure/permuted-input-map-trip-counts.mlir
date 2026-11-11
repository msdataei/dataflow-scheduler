// RUN: dataflow-scheduler-opt --reduction-loop-exposure --debug-only=reduction-loop-exposure --verify-each=false %s 2>&1 >/dev/null | FileCheck %s

// Verify that the trip count of each exposed reduction loop is the size of the
// loop dimension, read through the operand that indexes it, and not the size of
// the input axis with the same index.  The input map here is (d1, d2, d0): loop
// dims d1 and d2 are the reductions, of sizes 4 and 8, while input axes 1 and 2
// have sizes 8 and 64.
//
// Only the trip counts are checked.  The rest of the pass still slices the
// input and patches its transfer at the input axes numbered like the loop
// dims, which is only right for an identity input map, so the IR it builds for
// this map is not valid and per-pass verification is turned off.

// CHECK: reduction-loop-exposure: reduction_dims=[
// CHECK: 1 (size=4)
// CHECK: 2 (size=8)
// CHECK: ] total_reduction_size=32

#map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @permuted_input_map(%src: memref<4x8x64xf16, "L1">, %dst: memref<64xf16, "L1">) {
    ktdf.pipeline {
      %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#2) {
        ktdf.data_transfer from %src[0, 0, 0] size [4, 8, 64] to %p#0 size [2048] : memref<4x8x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%p#2) depends_out(%p#3) {
        %in = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 2048xf16> -> tensor<4x8x64xf16>
        %init = tensor.empty() : tensor<64xf16>
        %r = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%in : tensor<4x8x64xf16>) outs(%init : tensor<64xf16>) {
        ^bb0(%a: f16, %b: f16):
          %s = arith.addf %a, %b : f16
          linalg.yield %s : f16
        } -> tensor<64xf16>
        ktdf.write_to_fifo %r, %p#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#3) depends_out(none) {
        ktdf.data_transfer from %p#1 size [64] to %dst[0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
    return
  }
}
