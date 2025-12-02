linalg_transform = r"""
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %op, %loop = transform.structured.tile_using_for %0 tile_sizes [8] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)
    %peeled_op, %remainder = transform.loop.peel %loop {peel_front = false} : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)

    %opinner = transform.structured.match ops{["linalg.generic"]} in %peeled_op : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %opinner vector_sizes [8] : !transform.any_op


    %opremainder = transform.structured.match ops{["linalg.generic"]} in %remainder : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
"""
