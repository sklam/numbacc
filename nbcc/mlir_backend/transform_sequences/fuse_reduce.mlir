// This is trying to implement fusion for binop(A, broadcast(reduce(A)))
// However, the transformed loop is not fusing the binop and reduce.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%input: !transform.any_op) {
    // First find linalg.sub
    %reduce_ops = transform.structured.match ops{["linalg.reduce"]} in %input : (!transform.any_op) -> !transform.any_op
    %broadcast_ops = transform.structured.match ops{["linalg.broadcast"]} in %input : (!transform.any_op) -> !transform.any_op
    %sub_ops = transform.structured.match ops{["linalg.sub"]} in %input : (!transform.any_op) -> !transform.any_op

    // transform.print %reduce_ops : !transform.any_op
    // transform.print %broadcast_ops : !transform.any_op
    // transform.print %sub_ops : !transform.any_op

    %red1, %red2 = transform.split_handle %reduce_ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %bc1, %bc2 = transform.split_handle %broadcast_ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %sub1 = transform.split_handle %sub_ops : (!transform.any_op) -> !transform.any_op


    // transform.print %red1 : !transform.any_op
    // transform.print %bc1 : !transform.any_op
    // transform.print %sub1 : !transform.any_op

    %tiled_op, %forall_op = transform.structured.tile_using_forall %sub1 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)


    // transform.print %forall_op : !transform.any_op
    %fused_op2, %loop2 =  transform.structured.fuse_into_containing_op %bc1 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %red1 into %loop2 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)


    transform.yield
  }
}
