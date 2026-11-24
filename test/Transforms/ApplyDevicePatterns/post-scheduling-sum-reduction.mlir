// RUN: dataflow-scheduler-opt %s -allow-unregistered-dialect -apply-device-patterns='groups=post_scheduling' | FileCheck %s

// CHECK-LABEL:   func.func @sum_reduction() {
// CHECK-NEXT:     %[[ALLOC_0:.*]] = memref.alloc() : memref<64x64xf16, "L1">
// CHECK-NEXT:     %[[ALLOC_1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:     "test.simd_reduction_sum"(%[[ALLOC_0]], %[[ALLOC_1]]) : (memref<64x64xf16, "L1">, memref<64xf16, "L1">) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @sum_reduction_simd_lane_spans_two_axes() {
// CHECK-NEXT:     %[[ALLOC_2:.*]] = memref.alloc() : memref<64x32x2xf16, "L1">
// CHECK-NEXT:     %[[ALLOC_3:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:     "test.simd_reduction_sum"(%[[ALLOC_2]], %[[ALLOC_3]]) : (memref<64x32x2xf16, "L1">, memref<64xf16, "L1">) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @sum_reduction_simd_lane_mixes_reduction_and_parallel() {
// CHECK-NOT:       test.simd_reduction_sum
// CHECK:           linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK-NOT:       test.simd_reduction_sum
// CHECK:           return


// Verify that the post_scheduling PDL pattern in the sample device rewrites a
// linalg.generic sum-reduction (SIMD-lane, output via subview) into a bare
// test.simd_reduction op whose second operand is the subview's *source*.

#map_in  = affine_map<(d0, d1) -> (d0, d1)>
#map_out = affine_map<(d0, d1) -> (d0)>
#map_in3  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_out3 = affine_map<(d0, d1, d2) -> (d0)>
#map_in4  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map_out4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  func.func @sum_reduction() {
    %alloc_in  = memref.alloc() : memref<64x64xf16, "L1">
    %alloc_acc = memref.alloc() : memref<64xf16, "L1">

    // The pattern requires the output to be a memref.subview of the accumulator.
    %subview = memref.subview %alloc_acc[0][64][1]
        : memref<64xf16, "L1"> to memref<64xf16, "L1">

    linalg.generic {
      indexing_maps = [#map_in, #map_out],
      iterator_types = ["parallel", "reduction"]
    } ins(%alloc_in : memref<64x64xf16, "L1">)
      outs(%subview  : memref<64xf16, "L1">) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %out : f16
      linalg.yield %sum : f16
    }

    return
  }

  // A SIMD lane spanning two input axes: for f16 the sample device's compute unit
  // has 64 lanes, which the trailing [32, 2] axes hold together.  Both are
  // reductions, so the generic is still the SIMD-lane reduction.
  func.func @sum_reduction_simd_lane_spans_two_axes() {
    %alloc_in  = memref.alloc() : memref<64x32x2xf16, "L1">
    %alloc_acc = memref.alloc() : memref<64xf16, "L1">
    %subview = memref.subview %alloc_acc[0][64][1]
        : memref<64xf16, "L1"> to memref<64xf16, "L1">

    linalg.generic {
      indexing_maps = [#map_in3, #map_out3],
      iterator_types = ["parallel", "reduction", "reduction"]
    } ins(%alloc_in : memref<64x32x2xf16, "L1">)
      outs(%subview  : memref<64xf16, "L1">) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %out : f16
      linalg.yield %sum : f16
    }

    return
  }

  // A SIMD lane whose axes mix reduction and parallel: the 64-element SIMD lane
  // is the trailing [32, 2], and only the 2 is reduced, so each SIMD lane needs
  // 32 separate pair sums.  A whole-lane reduction would sum all 64 elements,
  // so the generic is not the SIMD-lane reduction and is left as it is.
  func.func @sum_reduction_simd_lane_mixes_reduction_and_parallel() {
    %alloc = memref.alloc() : memref<1x32x2xf16, "L1">
    %subview = memref.subview %alloc[0, 0, 0] [1, 32, 1] [1, 1, 1]
        : memref<1x32x2xf16, "L1">
          to memref<1x32x1xf16, strided<[64, 2, 1]>, "L1">

    linalg.generic {
      indexing_maps = [#map_in4, #map_out4],
      iterator_types = ["parallel", "parallel", "reduction", "parallel"]
    } ins(%alloc : memref<1x32x2xf16, "L1">)
      outs(%subview : memref<1x32x1xf16, strided<[64, 2, 1]>, "L1">) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %out : f16
      linalg.yield %sum : f16
    }

    return
  }
}
