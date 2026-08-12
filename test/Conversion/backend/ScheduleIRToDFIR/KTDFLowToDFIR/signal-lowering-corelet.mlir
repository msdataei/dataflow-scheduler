// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Test the corelet-aware signal lowering path.
//
// Case 1 (symmetric): PU owns two corelet units (MNILU-CL0, MNILU-CL1);
// signal peer also has one L1LU unit per corelet (L1LU-CL0, L1LU-CL1).
// The two L1LU operands must be merged into a single def_immutable_mapping
// keyed by iter_arg so CL0 only syncs with L1LU-CL0 and CL1 with L1LU-CL1.
//
// Case 2 (asymmetric): PU owns two corelet units (L1LU-CL0, L1LU-CL1);
// signal peer is MNILU — a single core-level unit with corelet = 0 that has
// no CL1 counterpart. The peer corelet set {0} does not cover the PU corelet
// set {0, 1}, so the lowering falls back to buildSignalQueryMap (core-index
// matching), producing [L1LU-CL0 -> MNILU, L1LU-CL1 -> MNILU].

// ---------------------------------------------------------------------------
// Case 1 checks
// ---------------------------------------------------------------------------
// CHECK-LABEL:   func.func private @signal_lowering_corelet_test() attributes {grid = [1]} {
// CHECK-NEXT:     %[[MNILU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-MNILU-CL0", type = "MNILU"} : index
// CHECK-NEXT:     %[[MNILU1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-MNILU-CL1", type = "MNILU"} : index
// CHECK-NEXT:     %[[L1LU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-L1LU-CL0", type = "L1LU"} : index
// CHECK-NEXT:     %[[L1LU1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-L1LU-CL1", type = "L1LU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[ARG:.*]] -> (%[[MNILU0]], %[[MNILU1]]) : {
// CHECK-NEXT:       %[[MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[MNILU0]] -> %[[L1LU0]]], {{\[}}%[[MNILU1]] -> %[[L1LU1]]]):index
// CHECK-NEXT:       %[[QUERY:.*]] = uniform.query_map(map:%[[MAP]], key:%[[ARG]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY]] : index
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// ---------------------------------------------------------------------------
// Case 2 checks
// ---------------------------------------------------------------------------
// CHECK-LABEL:   func.func private @signal_lowering_asymmetric_corelet_test() attributes {grid = [1]} {
// CHECK-NEXT:     %[[L1LU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-L1LU-CL0", type = "L1LU"} : index
// CHECK-NEXT:     %[[L1LU1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-L1LU-CL1", type = "L1LU"} : index
// CHECK-NEXT:     %[[MNILU:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[ARG2:.*]] -> (%[[L1LU0]], %[[L1LU1]]) : {
// CHECK-NEXT:       %[[MAP2:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU0]] -> %[[MNILU]]], {{\[}}%[[L1LU1]] -> %[[MNILU]]]):index
// CHECK-NEXT:       %[[QUERY2:.*]] = uniform.query_map(map:%[[MAP2]], key:%[[ARG2]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY2]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY2]] : index
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // Case 1: symmetric — peer has one L1LU unit per PU corelet.
  func.func private @signal_lowering_corelet_test() attributes {grid = [1]} {
    %mnilu0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-MNILU-CL0", type = "MNILU"} : index
    %mnilu1 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-MNILU-CL1", type = "MNILU"} : index
    %l1lu0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-L1LU-CL0", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-L1LU-CL1", type = "L1LU"} : index
    %tile_id = ktdp.get_compute_tile_id : index

    // Multi-key maps keyed by corelet index so both units are reachable.
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mnilu_map = uniform.def_immutable_mapping([%c0 -> %mnilu0], [%c1 -> %mnilu1]):index
    %mnilu_query = uniform.query_map(map:%mnilu_map, key:%tile_id) : index

    %c0_l = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %l1lu_map = uniform.def_immutable_mapping([%c0_l -> %l1lu0], [%c1_l -> %l1lu1]):index
    %l1lu_query = uniform.query_map(map:%l1lu_map, key:%tile_id) : index

    ktdf_lowering.execute_on %mnilu_query {
      ktdf_lowering.signal %mnilu_query, %l1lu_query
    }

    return
  }

  // Case 2: asymmetric — PU owns L1LU-CL0/CL1; peer is a single MNILU unit
  // (corelet = 0 only). buildSignalQueryMap matches by core index so both
  // L1LU corelets (core = 0) map to the same MNILU.
  func.func private @signal_lowering_asymmetric_corelet_test() attributes {grid = [1]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-L1LU-CL0", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-L1LU-CL1", type = "L1LU"} : index
    %mnilu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %tile_id = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %l1lu_map = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]):index
    %l1lu_query = uniform.query_map(map:%l1lu_map, key:%tile_id) : index

    %c0_m = arith.constant 0 : index
    %mnilu_map = uniform.def_immutable_mapping([%c0_m -> %mnilu]):index
    %mnilu_query = uniform.query_map(map:%mnilu_map, key:%tile_id) : index

    ktdf_lowering.execute_on %l1lu_query {
      ktdf_lowering.signal %l1lu_query, %mnilu_query
    }

    return
  }
}
