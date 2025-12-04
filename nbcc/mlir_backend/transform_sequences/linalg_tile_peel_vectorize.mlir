// MLIR Transform Sequence: Linalg Tiling, Peeling, and Vectorization
//
// This transformation sequence optimizes linalg.generic operations through a
// three-stage process: tiling, peeling, and vectorization.
//
// Transformation Pipeline:
// 1. Tile linalg.generic operations with tile size of 8
// 2. Peel the resulting loops to handle remainder iterations
// 3. Vectorize the peeled operations with vector size of 8

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // Step 1: Find all linalg.generic operations in the input
    %matches = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Step 2: Apply tiling, peeling, and vectorization to each matched operation
    transform.foreach %matches : !transform.any_op {
        ^bb0(%0: !transform.any_op):
        // Tile the operation with tile size 8, creating a loop structure
        %op, %loop = transform.structured.tile_using_for %0 tile_sizes [8] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)

        // Peel the loop to separate the main iterations from remainder
        // peel_front=false means we peel from the end (remainder at the end)
        %peeled_op, %remainder = transform.loop.peel %loop {peel_front = false} : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)

        // Find linalg.generic operations within the peeled main loop
        %opinner = transform.structured.match ops{["linalg.generic"]} in %peeled_op : (!transform.any_op) -> !transform.any_op

        // Vectorize the main loop iterations with vector size 8
        transform.structured.vectorize %opinner vector_sizes [8] : !transform.any_op
    }

    transform.yield
  }
}