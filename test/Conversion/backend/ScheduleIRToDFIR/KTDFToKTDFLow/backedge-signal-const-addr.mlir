// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdf-to-ktdflowering)"  -allow-unregistered-dialect %s | FileCheck %s

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
// CHECK:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:       %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
// CHECK-NEXT:       %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK-NEXT:       %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu-CL0", type = "l1lu"} : index
// CHECK-NEXT:       %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-l1lu-CL1", type = "l1lu"} : index
// CHECK-NEXT:       %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu-CL0", type = "sfu"} : index
// CHECK-NEXT:       %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfu-CL1", type = "sfu"} : index
// CHECK-NEXT:       %[[GET_UNIT_6:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su-CL0", type = "l1su"} : index
// CHECK-NEXT:       %[[GET_UNIT_7:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-l1su-CL1", type = "l1su"} : index
// CHECK-NEXT:       %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_0]] -> %[[GET_UNIT_0]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_2]] -> %[[GET_UNIT_2]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_3]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_4:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_4]] -> %[[GET_UNIT_4]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_4:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_4]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_5:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_5]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_5:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_5]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_6:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_6]] -> %[[GET_UNIT_6]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_6:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_6]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_7:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_7]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_7:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_7]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_8:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:       %[[CONSTANT_9:.*]] = arith.constant 64 : index
// CHECK-NEXT:       %[[CONSTANT_10:.*]] = arith.constant 63 : index
// CHECK-NEXT:       %[[CONSTANT_11:.*]] = arith.constant 4 : index
// CHECK-NEXT:       %[[CONSTANT_12:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_13:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_14:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTANT_15:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_12]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_2]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_14]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2x64xf16> to memref<2x64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, #ktdp.memory_space<global>> to memref<2x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<2x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>> to memref<2x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:       ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_1]] {
// CHECK-NEXT:         %[[CONSTANT_16:.*]] = arith.constant 0 : index
// CHECK-NEXT:         %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_16]] : index to memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:         %[[CONSTANT_17:.*]] = arith.constant 65536 : index
// CHECK-NEXT:         %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_17]] : index to memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK-NEXT:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_12]] to %[[CONSTANT_15]] step %[[CONSTANT_13]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[VAL_0]], 0, %[[CONSTANT_12]] * 64] size [1, 256, 64] to %[[UNREALIZED_CONVERSION_CAST_0]]{{\[}}%[[VAL_0]], 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf_lowering.signal %[[QUERY_MAP_0]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]]
// CHECK-NEXT:         ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK-NEXT:           ktdf.parallel (%[[VAL_1:.*]], %[[VAL_2:.*]]) = (%[[CONSTANT_12]]) to (%[[CONSTANT_15]]) step (%[[CONSTANT_13]]) distribute(num_instances = 2) {
// CHECK-NEXT:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_12]] to %[[CONSTANT_11]] step %[[CONSTANT_13]] {
// CHECK-NEXT:               %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_3]], %[[CONSTANT_12]] : index
// CHECK-NEXT:               ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK-NEXT:                 %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                 %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                 %[[FIFO_2:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                 %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]] {
// CHECK-NEXT:                   %[[CMPI_1:.*]] = arith.cmpi ne, %[[VAL_3]], %[[CONSTANT_12]] : index
// CHECK-NEXT:                   scf.if %[[CMPI_1]] {
// CHECK-NEXT:                     ktdf_lowering.signal %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]]
// CHECK-NEXT:                   }
// CHECK-NEXT:                   scf.if %[[CMPI_0]] {
// CHECK-NEXT:                   } else {
// CHECK-NEXT:                     ktdf.data_transfer from %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_1]], %[[CONSTANT_12]], %[[CONSTANT_12]]] size [1, 1, 64] to %[[FIFO_1]] size [64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                   }
// CHECK-NEXT:                   scf.for %[[VAL_4:.*]] = %[[CONSTANT_12]] to %[[CONSTANT_9]] step %[[CONSTANT_13]] {
// CHECK-NEXT:                     ktdf.data_transfer from %[[UNREALIZED_CONVERSION_CAST_0]]{{\[}}%[[VAL_1]], %[[CONSTANT_12]], %[[VAL_3]] * 64 + %[[VAL_4]], %[[CONSTANT_12]]] size [1, 1, 1, 64] to %[[FIFO_0]] size [64] : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 ktdf_lowering.execute_on %[[QUERY_MAP_4]], %[[QUERY_MAP_5]] {
// CHECK-NEXT:                   %[[CONSTANT_18:.*]] = arith.constant 0 : index
// CHECK-NEXT:                   %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_18]] : index to memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:                   linalg.fill ins(%[[CONSTANT_8]] : f16) outs(%[[UNREALIZED_CONVERSION_CAST_2]] : memref<1x64xf16, "SFU_REG">)
// CHECK-NEXT:                   scf.for %[[VAL_5:.*]] = %[[CONSTANT_12]] to %[[CONSTANT_9]] step %[[CONSTANT_13]] {
// CHECK-NEXT:                     %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[FIFO_0]] : <"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
// CHECK-NEXT:                     linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : memref<1x1x64xf16>) outs(%[[UNREALIZED_CONVERSION_CAST_2]] : memref<1x64xf16, "SFU_REG">) {
// CHECK-NEXT:                     ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK-NEXT:                       %[[ADDF_0:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f16
// CHECK-NEXT:                       linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                     }
// CHECK-NEXT:                     %[[CMPI_2:.*]] = arith.cmpi eq, %[[VAL_5]], %[[CONSTANT_10]] : index
// CHECK-NEXT:                     scf.if %[[CMPI_2]] {
// CHECK-NEXT:                       ktdf.write_to_fifo %[[UNREALIZED_CONVERSION_CAST_2]], %[[FIFO_2]] : memref<1x64xf16, "SFU_REG">, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 ktdf_lowering.execute_on %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK-NEXT:                   scf.for %[[VAL_8:.*]] = %[[CONSTANT_12]] to %[[CONSTANT_9]] step %[[CONSTANT_13]] {
// CHECK-NEXT:                     %[[CMPI_3:.*]] = arith.cmpi eq, %[[VAL_8]], %[[CONSTANT_10]] : index
// CHECK-NEXT:                     scf.if %[[CMPI_3]] {
// CHECK-NEXT:                       ktdf.data_transfer from %[[FIFO_2]] size [64] to %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_1]], %[[CONSTANT_12]], %[[CONSTANT_12]]] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                   %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_11]], %[[CONSTANT_13]] : index
// CHECK-NEXT:                   %[[CMPI_4:.*]] = arith.cmpi ne, %[[VAL_3]], %[[SUBI_0]] : index
// CHECK-NEXT:                   scf.if %[[CMPI_4]] {
// CHECK-NEXT:                     ktdf_lowering.signal %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]]
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf.parallel_yield
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf_lowering.signal %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_1]]
// CHECK-NEXT:         ktdf_lowering.execute_on %[[QUERY_MAP_1]] {
// CHECK-NEXT:           scf.for %[[VAL_9:.*]] = %[[CONSTANT_12]] to %[[CONSTANT_15]] step %[[CONSTANT_13]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[UNREALIZED_CONVERSION_CAST_1]]{{\[}}%[[VAL_9]], 0, 0] size [1, 1, 64] to %[[CAST_1]]{{\[}}%[[VAL_9]], %[[CONSTANT_12]] * 64] size [1, 64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<2x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
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
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c63 = arith.constant 63 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c2 = arith.constant 2 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>>
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2x64xf16> to memref<2x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, #ktdp.memory_space<global>> to memref<2x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>>
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<2x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>> to memref<2x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %c0_3 = arith.constant 0 : index
          %3 = builtin.unrealized_conversion_cast %c0_3 : index to memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>
          %c65536 = arith.constant 65536 : index
          %4 = builtin.unrealized_conversion_cast %c65536 : index to memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
          %5 = ktdf.create_token : !ktdf.token
          %6 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %3, %4, %5, %6 : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            ktdf.data_transfer from %cast[%arg0, 0, %c0 * 64] size [1, 256, 64] to %2#0[%arg0, 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          ktdf.parallel (%arg0, %arg1) = (%c0) to (%c2) step (%c1) distribute(num_instances = 2) {
            scf.for %arg2 = %c0 to %c4 step %c1 {
              %3 = arith.cmpi eq, %arg2, %c0 : index
              ktdf.pipeline {
                %4:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                  %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  %6 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  %7 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                  %8 = ktdf.create_token : !ktdf.token
                  %9 = ktdf.create_token : !ktdf.token
                  ktdf.private_yield %5, %6, %7, %8, %9 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
                }
                ktdf.stage depends_in(none) depends_out(%4#3) {
                  scf.if %3 {
                  } else {
                    ktdf.data_transfer from %2#1[%arg0, %c0, %c0] size [1, 1, 64] to %4#1 size [64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  }
                  scf.for %arg3 = %c0 to %c64 step %c1 {
                    ktdf.data_transfer from %2#0[%arg0, %c0, %arg2 * 64 + %arg3, %c0] size [1, 1, 1, 64] to %4#0 size [64] : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                } {applicable_units = ["L1LU"]}
                ktdf.stage depends_in(%4#3) depends_out(%4#4) {
                  %c0_3 = arith.constant 0 : index
                  %5 = builtin.unrealized_conversion_cast %c0_3 : index to memref<1x64xf16, "SFU_REG">
                  linalg.fill ins(%cst : f16) outs(%5 : memref<1x64xf16, "SFU_REG">)
                  scf.for %arg3 = %c0 to %c64 step %c1 {
                    %6 = ktdf.read_from_fifo %4#0 : <"L1LU" -> "SFU", 64xf16> -> memref<1x1x64xf16>
                    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%6 : memref<1x1x64xf16>) outs(%5 : memref<1x64xf16, "SFU_REG">) {
                    ^bb0(%in: f16, %out: f16):
                      %8 = arith.addf %in, %out : f16
                      linalg.yield %8 : f16
                    }
                    %7 = arith.cmpi eq, %arg3, %c63 : index
                    scf.if %7 {
                      ktdf.write_to_fifo %5, %4#2 : memref<1x64xf16, "SFU_REG">, <"SFU" -> "L1SU", 64xf16>
                    }
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                } {applicable_units = ["SFU"]}
                ktdf.stage depends_in(%4#4) depends_out(none) {
                  scf.for %arg3 = %c0 to %c64 step %c1 {
                    %5 = arith.cmpi eq, %arg3, %c63 : index
                    scf.if %5 {
                      ktdf.data_transfer from %4#2 size [64] to %2#1[%arg0, %c0, %c0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
                    }
                  } {loop_type = #ktdf.loop_type<reduction_loop>}
                } {applicable_units = ["L1SU"]}
              }
            }
            ktdf.parallel_yield
          }
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            ktdf.data_transfer from %2#1[%arg0, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<2x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}

