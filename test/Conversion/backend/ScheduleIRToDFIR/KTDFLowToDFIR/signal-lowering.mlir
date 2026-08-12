// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s


// CHECK-LABEL:   func.func private @signal_lowering_test() attributes {grid = [2]} {
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_0]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY_MAP_0]] : index
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_1:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_1]]) : index
// CHECK-NEXT:       dataflow.sync_send %[[QUERY_MAP_1]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT:       dataflow.sync_recv %[[QUERY_MAP_1]] : index
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }








module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func private @signal_lowering_test() attributes {grid = [2]} {
    %mnilu0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %mnilu1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %tile_id = ktdp.get_compute_tile_id : index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mnilu_map = uniform.def_immutable_mapping([%c0 -> %mnilu0], [%c1 -> %mnilu1]):index
    %mnilu_query = uniform.query_map(map:%mnilu_map, key:%tile_id) : index

    %c0_0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %l1lu_map = uniform.def_immutable_mapping([%c0_0 -> %l1lu0], [%c1_0 -> %l1lu1]):index
    %l1lu_query = uniform.query_map(map:%l1lu_map, key:%tile_id) : index

    ktdf_lowering.execute_on %mnilu_query {
      ktdf_lowering.signal %mnilu_query, %l1lu_query
    }

    ktdf_lowering.execute_on %l1lu_query {
      ktdf_lowering.signal %l1lu_query, %mnilu_query
    }

    return
  }
}
