// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Verify that a memref.copy whose source is a ktdf.read_from_fifo is lowered
// to an agen.vector_store that writes the received vector directly into the
// destination buffer, with no intervening memref.copy surviving.

// CHECK: #[[$STORE_ORDER:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$STORE_SET:.+]]   = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL: func.func @memref_copy_from_fifo
// CHECK:         dataflow.program_unit
// CHECK:           %[[RECV:.+]] = dataflow.receive
// CHECK-NEXT:      agen.vector_store %[[RECV]]
// CHECK-SAME:        {store_order = #[[$STORE_ORDER]], store_set = #[[$STORE_SET]]}
// CHECK-NOT:       memref.copy

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @memref_copy_from_fifo() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>

    // Producer: sends one 1x64 slice through the FIFO.
    ktdf_lowering.execute_on %u_l1lu {
      %src = memref.alloc() : memref<1x64xf16, "L1">
      ktdf.data_transfer from %src[%c0, %c0] size [1, 64] to %fifo size [64]
        : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }

    // Consumer: read_from_fifo produces a memref, then memref.copy moves it
    // into an SFU register buffer.  The copy must be replaced by
    // agen.vector_store without surviving as a memref.copy.
    ktdf_lowering.execute_on %u_sfu {
      %reg = memref.alloc() : memref<1x64xf16, "SFU_REG">
      %read = ktdf.read_from_fifo %fifo
        : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> memref<1x64xf16>
      memref.copy %read, %reg : memref<1x64xf16> to memref<1x64xf16, "SFU_REG">
    }

    return
  }
}
