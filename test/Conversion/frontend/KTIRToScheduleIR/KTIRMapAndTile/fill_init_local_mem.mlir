// RUN: dataflow-scheduler-opt -pass-pipeline='builtin.module(builtin.module(func.func(ktir-map-and-tile)))' %s

// A reduction initialised by a linalg.fill of a scalar loaded from local
// memory is not supported yet: the one-element load's use is a
// tensor.extract, not the tensor.extract_slice tiling leaves, and lowering it
// fails with "no `tensor.extract_slice` sink".
// XFAIL: *

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0)>
module {
  module {
    func.func @max_onstick_local_init() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c8589934592 = arith.constant 8589934592 : index
      %c1024 = arith.constant 1024 : index
      %c0 = arith.constant 0 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 32], strides: [8192, 32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x32xf32>
      %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<2x256x32xf32> -> !ktdp.access_tile<2x256x32xindex>
      %2 = ktdp.construct_memory_view %c1024, sizes: [256, 32], strides: [32, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<ct_local>} : memref<256x32xf32, #ktdp.memory_space<ct_local>>
      %3 = ktdp.construct_access_tile %2[%c0, %c0] {access_tile_order = #map1, access_tile_set = #set2} : memref<256x32xf32, #ktdp.memory_space<ct_local>> -> !ktdp.access_tile<1x1xindex>
      %4 = ktdp.load %3 : <1x1xindex> -> tensor<1x1xf32>
      %extracted = tensor.extract %4[%c0, %c0] : tensor<1x1xf32>
      %5 = tensor.empty() : tensor<256x32xf32>
      %6 = ktdp.load %1 : <2x256x32xindex> -> tensor<2x256x32xf32>
      %7 = linalg.fill ins(%extracted : f32) outs(%5 : tensor<256x32xf32>) -> tensor<256x32xf32>
      %8 = ktdp.construct_memory_view %c8589934592, sizes: [256, 32], strides: [32, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<256x32xf32>
      %9 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%6 : tensor<2x256x32xf32>) outs(%7 : tensor<256x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        %11 = arith.maximumf %in, %out : f32
        linalg.yield %11 : f32
      } -> tensor<256x32xf32>
      %10 = ktdp.construct_access_tile %8[%c0, %c0] {access_tile_order = #map1, access_tile_set = #set1} : memref<256x32xf32> -> !ktdp.access_tile<256x32xindex>
      ktdp.store %9, %10 : tensor<256x32xf32>, <256x32xindex>
      return
    }
  }
}
