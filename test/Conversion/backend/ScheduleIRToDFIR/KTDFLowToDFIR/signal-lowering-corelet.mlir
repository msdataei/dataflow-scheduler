// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Test the corelet-aware signal lowering path.
// The PU owns two corelet units of the same resource type (MNILU-CL0 and
// MNILU-CL1).  The signal carries one query_map per corelet of the peer type
// (L1LU-CL0, L1LU-CL1).  The two L1LU operands must be merged into a single
// def_immutable_mapping + query_map keyed by iter_arg so that at runtime
// CL0 only syncs with L1LU-CL0 and CL1 only syncs with L1LU-CL1 -
// not a cross-corelet sync.

// CHECK-LABEL:   func.func private @signal_lowering_corelet_test() attributes {grid = [1]} {
// CHECK-NEXT:     %[[MNILU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-MNILU-CL0", type = "MNILU"} : index
// CHECK-NEXT:     %[[MNILU1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-MNILU-CL1", type = "MNILU"} : index
// CHECK-NEXT:     %[[L1LU0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-L1LU-CL0", type = "L1LU"} : index
// CHECK-NEXT:     %[[L1LU1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-L1LU-CL1", type = "L1LU"} : index
// CHECK:          dataflow.program_unit iter_arg : %[[ARG:.*]] -> (%[[MNILU0]], %[[MNILU1]]) : {
// CHECK-NEXT:       %[[MAP:.*]] = uniform.def_immutable_mapping({{\[}}%[[MNILU0]] -> %[[L1LU0]]], {{\[}}%[[MNILU1]] -> %[[L1LU1]]]):index
// CHECK-NEXT:       %[[QUERY:.*]] = uniform.query_map(map:%[[MAP]], key:%[[ARG]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY]] : index
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func private @signal_lowering_corelet_test() attributes {grid = [1]} {
    %mnilu0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-MNILU-CL0", type = "MNILU"} : index
    %mnilu1 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-MNILU-CL1", type = "MNILU"} : index
    %l1lu0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-L1LU-CL0", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-L1LU-CL1", type = "L1LU"} : index
    %tile_id = ktdp.get_compute_tile_id : index

    // One query_map per MNILU corelet (current unit type), keyed by tile_id.
    %c0_a = arith.constant 0 : index
    %mnilu0_map = uniform.def_immutable_mapping([%c0_a -> %mnilu0]):index
    %mnilu0_query = uniform.query_map(map:%mnilu0_map, key:%tile_id) : index

    %c0_b = arith.constant 0 : index
    %mnilu1_map = uniform.def_immutable_mapping([%c0_b -> %mnilu1]):index
    %mnilu1_query = uniform.query_map(map:%mnilu1_map, key:%tile_id) : index

    // One query_map per L1LU corelet (peer unit type), keyed by tile_id.
    %c0_c = arith.constant 0 : index
    %l1lu0_map = uniform.def_immutable_mapping([%c0_c -> %l1lu0]):index
    %l1lu0_query = uniform.query_map(map:%l1lu0_map, key:%tile_id) : index

    %c0_d = arith.constant 0 : index
    %l1lu1_map = uniform.def_immutable_mapping([%c0_d -> %l1lu1]):index
    %l1lu1_query = uniform.query_map(map:%l1lu1_map, key:%tile_id) : index

    // PU owns both MNILU corelets; signal includes both L1LU corelets.
    // Expected: one merged map [MNILU-CL0->L1LU-CL0, MNILU-CL1->L1LU-CL1]
    // queried by iter_arg -> one sync_send + sync_recv (not two pairs).
    dataflow.program_unit iter_arg : %arg0 -> (%mnilu0, %mnilu1) : {
      ktdf_lowering.signal %mnilu0_query, %mnilu1_query, %l1lu0_query, %l1lu1_query
    }

    return
  }
}
