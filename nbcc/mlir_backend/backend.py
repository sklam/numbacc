from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import mlir.dialects.arith as arith
import mlir.dialects.cf as cf
import mlir.dialects.func as func
import mlir.dialects.scf as scf

from mlir.dialects import llvm
import mlir.execution_engine as execution_engine
import mlir.ir as ir
import mlir.passmanager as passmanager
import mlir.runtime as runtime
import numpy as np
from sealir import ase
from sealir.rvsdg import grammar as rg
from sealir.rvsdg import internal_prefix

from ..egraph import grammar as sg


# ## MLIR Backend Implementation
#
# Define the core MLIR backend class that handles type lowering and
# expression compilation.

_DEBUG = True


@dataclass(frozen=True)
class LowerStates(ase.TraverseState):
    push: Callable
    get_region_args: Callable
    function_block: func.FuncOp
    constant_block: ir.Block


class Backend:
    def __init__(self):
        self.context = context = ir.Context()
        # context.allow_unregistered_dialects = True
        self._declared = {}
        with context:
            self.f32 = ir.F32Type.get(context=context)
            self.f64 = ir.F64Type.get(context=context)
            self.i8 = ir.IntegerType.get_signless(8, context=context)
            self.i32 = ir.IntegerType.get_signless(32, context=context)
            self.i64 = ir.IntegerType.get_signless(64, context=context)
            self.boolean = ir.IntegerType.get_signless(1, context=context)
            self.io_type = ir.IntegerType.get_signless(1, context=context)
            self.llvm_ptr = ir.Type.parse("!llvm.ptr")

    def lower_type(self, ty: NbOp_Type):
        """Type Lowering

        Convert SealIR types to MLIR types for compilation.
        """
        match ty:
            case NbOp_Type("Int64"):
                return self.i64
            case NbOp_Type("Float64"):
                return self.f64
            case NbOp_Type("Float32"):
                return self.f32
        raise NotImplementedError(f"unknown type: {ty}")

    def get_return_types(self, root):
        return
        # return (
        #     self.lower_type(
        #         Attributes(root.body.begin.attrs).get_return_type(root.body)
        #     ),
        # )

    def lower(self, root: rg.Func, argtypes):
        """Expression Lowering

        Lower RVSDG expressions to MLIR operations, handling control flow
        and data flow constructs.
        """
        context = self.context
        self.loc = loc = ir.Location.name(f"{self}.lower()", context=context)
        self.module = module = ir.Module.create(loc=loc)

        function_name = root.fname

        # Get the module body pointer so we can insert content into the
        # module.
        self.module_body = module_body = ir.InsertionPoint(module.body)
        # Convert SealIR types to MLIR types.
        input_types = tuple([self.lower_type(x) for x in argtypes])
        output_types = self.get_return_types(root)

        with context, loc, module_body:
            # Constuct a function that emits a callable C-interface.
            fnty = func.FunctionType.get([], [])
            fun = func.FuncOp(function_name, fnty)
            fun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

            # Define two blocks within the function, a constant block to
            # define all the constants and a function block for the
            # actual content. This is done to prevent non-dominant use
            # of constants. (Use of a constant when declaration is done in
            # a region that isn't initialized.)
            const_block = fun.add_entry_block()
            fun.body.blocks.append(*[], arg_locs=None)
            func_block = fun.body.blocks[1]

        # Define entry points of both the blocks.
        constant_entry = ir.InsertionPoint(const_block)
        function_entry = ir.InsertionPoint(func_block)

        region_args = []

        @contextmanager
        def push(arg_values):
            region_args.append(tuple(arg_values))
            try:
                yield
            finally:
                region_args.pop()

        def get_region_args():
            return region_args[-1]

        with context, loc, function_entry:
            memo = ase.traverse(
                root,
                self.lower_expr,
                LowerStates(
                    push=push,
                    get_region_args=get_region_args,
                    function_block=fun,
                    constant_block=constant_entry,
                ),
            )

        # Use a break to jump from the constant block to the function block.
        # note that this is being inserted at end of constant block after the
        # Function construction when all the constants have been initialized.
        with context, loc, constant_entry:
            cf.br([], fun.body.blocks[1])

        return module

    def run_passes(self, module):
        """MLIR Pass Pipeline

        Apply MLIR passes for optimization and lowering to LLVM IR.
        """
        if _DEBUG:
            module.dump()

        if _DEBUG:
            module.context.enable_multithreading(False)

        pass_man = passmanager.PassManager(context=module.context)
        pass_man.add("convert-linalg-to-loops")
        pass_man.add("convert-scf-to-cf")
        pass_man.add("finalize-memref-to-llvm")
        pass_man.add("convert-math-to-libm")
        pass_man.add("convert-func-to-llvm")
        pass_man.add("convert-arith-to-llvm")
        pass_man.add("convert-cf-to-llvm")
        pass_man.add("convert-index-to-llvm")
        pass_man.add("reconcile-unrealized-casts")
        pass_man.enable_verifier(True)
        pass_man.run(module.operation)
        # Output LLVM-dialect MLIR
        if _DEBUG:
            module.dump()
        return module

    def _cast_return_value(self, val):
        return val

    def lower_expr(self, expr: SExpr, state: LowerStates):
        """Expression Lowering Implementation

        Implement the core expression lowering logic for various RVSDG
        constructs including functions, regions, control flow, and operations.
        """

        module = self.module
        context = self.context
        match expr:
            case rg.Func(args=args, body=body):
                names = {
                    argspec.name: state.function_block.arguments[i]
                    for i, argspec in enumerate(args.arguments)
                }
                argvalues = []
                for k in body.begin.inports:
                    if k == internal_prefix("io"):
                        v = arith.constant(self.io_type, 0)
                    else:
                        v = names[k]
                    argvalues.append(v)

                with state.push(argvalues):
                    outs = yield body

                portnames = [p.name for p in body.ports]
                try:
                    retidx = portnames.index(internal_prefix("ret"))
                except ValueError as e:
                    assert "!ret" in str(e)
                    func.ReturnOp([])
                else:
                    retval = outs[retidx]
                    func.ReturnOp([self._cast_return_value(retval)])
            case rg.RegionBegin(inports=ins):
                portvalues = []
                for i, k in enumerate(ins):
                    pv = state.get_region_args()[i]
                    portvalues.append(pv)
                return tuple(portvalues)

            case rg.RegionEnd(
                begin=rg.RegionBegin() as begin,
                ports=ports,
            ):
                yield begin
                portvalues = []
                for p in ports:
                    pv = yield p.value
                    portvalues.append(pv)
                return tuple(portvalues)

            case rg.ArgRef(idx=int(idx), name=str(name)):
                return state.function_block.arguments[idx]

            case rg.Unpack(val=source, idx=int(idx)):
                ports = yield source
                return ports[idx]

            case rg.DbgValue(value=value):
                val = yield value
                return val

            case rg.PyInt(int(ival)):
                with state.constant_block:
                    const = arith.constant(self.i32, ival)  # HACK: select type
                return const

            case rg.PyBool(int(ival)):
                with state.constant_block:
                    const = arith.constant(self.boolean, ival)
                return const

            case rg.PyFloat(float(fval)):
                with state.constant_block:
                    const = arith.constant(self.f64, fval)
                return const

            case rg.PyStr(str(strval)):
                with self.module_body:
                    encoded = strval.encode("utf8")
                    length = len(encoded)

                    struct_type = ir.Type.parse(
                        f"!llvm.struct<(i64, array<{length} x i8>)>"
                    )
                    struct_value = struct_value = ir.ArrayAttr.get(
                        [
                            ir.IntegerAttr.get(self.i64, length),
                            ir.StringAttr.get(encoded),
                        ]
                    )

                    sym_name = ".const.str" + str(hash(expr))
                    llvm.GlobalOp(
                        global_type=struct_type,
                        sym_name=sym_name,
                        linkage=ir.Attribute.parse("#llvm.linkage<private>"),
                        constant=True,
                        value=struct_value,
                        addr_space=0,
                    )
                with state.constant_block:
                    ptr_type = self.llvm_ptr
                    str_addr = llvm.AddressOfOp(
                        ptr_type, ir.FlatSymbolRefAttr.get(sym_name)
                    )

                return str_addr

            # NBCC specific
            case sg.BuiltinOp("i32_add", (lhs, rhs)):
                lhs = yield lhs
                rhs = yield rhs
                return arith.addi(lhs, rhs)

            case sg.BuiltinOp("i32_sub", (lhs, rhs)):
                lhs = yield lhs
                rhs = yield rhs
                return arith.subi(lhs, rhs)

            case sg.BuiltinOp("i32_gt", (lhs, rhs)):
                lhs = yield lhs
                rhs = yield rhs
                return arith.cmpi(arith.CmpIPredicate.sgt, lhs, rhs)

            case sg.BuiltinOp("print_i32", (io, operand)):
                io = yield io
                operand = yield operand

                print_fn = self.declare_builtins(
                    "spy_builtins$print_i32", [self.i32], []
                )
                func.call(
                    print_fn.type.results, "spy_builtins$print_i32", [operand]
                )
                return io

            case sg.BuiltinOp("print_str", (io, operand)):
                io = yield io
                operand = yield operand

                print_fn = self.declare_builtins(
                    "spy_builtins$print_str", [self.llvm_ptr], []
                )

                func.call(
                    print_fn.type.results, "spy_builtins$print_str", [operand]
                )
                return io

            case sg.BuiltinOp("struct_make", args=raw_args):
                args = []
                for v in raw_args:
                    args.append((yield v))

                tys = [v.type for v in args]
                struct_type = llvm.StructType.get_literal(tys)

                struct_value = llvm.UndefOp(struct_type)
                for i, v in enumerate(args):
                    struct_value = llvm.insertvalue(
                        struct_value, v, ir.DenseI64ArrayAttr.get([i])
                    )
                return struct_value

            case sg.BuiltinOp("struct_get", args=(struct, int(pos))):
                struct_value = yield struct

                resty = self.i32  # HACK
                return llvm.extractvalue(
                    resty, struct_value, ir.DenseI64ArrayAttr.get([pos])
                )

            # case NbOp_Gt_Int64(lhs, rhs):
            #     lhs = yield lhs
            #     rhs = yield rhs
            #     return arith.cmpi(4, lhs, rhs)

            # case NbOp_Add_Int64(lhs, rhs):
            #     lhs = yield lhs
            #     rhs = yield rhs
            #     return arith.addi(lhs, rhs)

            # case NbOp_Sub_Int64(lhs, rhs):
            #     lhs = yield lhs
            #     rhs = yield rhs
            #     return arith.subi(lhs, rhs)

            # case NbOp_Add_Float64(lhs, rhs):
            #     lhs = yield lhs
            #     rhs = yield rhs
            #     return arith.addf(lhs, rhs)
            # case NbOp_Sub_Float64(lhs, rhs):
            #     lhs = yield lhs
            #     rhs = yield rhs
            #     return arith.subf(lhs, rhs)
            # case NbOp_Lt_Int64(lhs, rhs):
            #     lhs = yield lhs
            #     rhs = yield rhs
            #     return arith.cmpi(2, lhs, rhs)
            # case NbOp_Sub_Int64(lhs, rhs):
            #     lhs = yield lhs
            #     rhs = yield rhs
            #     return arith.subi(lhs, rhs)

            # case NbOp_CastI64ToF64(operand):
            #     val = yield operand
            #     return arith.sitofp(self.f64, val)
            # case NbOp_Div_Int64(lhs, rhs):
            #     lhs = yield lhs
            #     rhs = yield rhs

            #     return arith.divf(
            #         arith.sitofp(self.f64, lhs), arith.sitofp(self.f64, rhs)
            #     )
            # ##### more
            # case NbOp_Not_Int64(operand):
            #     # Implement unary not
            #     opval = yield operand
            #     return arith.cmpi(0, opval, arith.constant(self.i64, 0))
            case rg.PyBool(val):
                return arith.constant(self.boolean, val)

            case rg.PyInt(val):
                return arith.constant(self.i64, val)

            case rg.IfElse(
                cond=cond, body=body, orelse=orelse, operands=operands
            ):
                condval = yield cond
                operand_vals = []
                for op in operands:
                    operand_vals.append((yield op))

                result_tys: list[ir.Type] = []

                # MLIR Workaround: We need to create detached blocks first to
                # build the then/else bodies to know about the MLIR types.

                with state.push(operand_vals):
                    # Make a detached module to temporarily house the blocks
                    fake = ir.Module.create()
                    then_block = fake.body.create_after()
                    with ir.InsertionPoint(then_block):
                        value_body = yield body
                        scf.YieldOp([x for x in value_body])
                        result_tys.extend(x.type for x in value_body)

                    else_block = then_block.create_after()
                    with ir.InsertionPoint(else_block):
                        value_else = yield orelse
                        scf.YieldOp([x for x in value_else])
                        for x, expected in zip(
                            value_else, result_tys, strict=True
                        ):
                            assert x.type == expected

                # Build the MLIR If-else
                if_op = scf.IfOp(
                    cond=condval, results_=result_tys, hasElse=True
                )

                # Move operations from detached then_block
                # to actual IfOp then_block
                with ir.InsertionPoint(if_op.then_block):
                    zero = arith.constant(self.i32, 0)
                    insertpt = arith.OrIOp(zero, zero)

                    for op in list(then_block.operations):
                        op.move_after(insertpt)
                        insertpt = op

                # Move operations from detached else_block
                # to actual IfOp else_block
                with ir.InsertionPoint(if_op.else_block):
                    zero = arith.constant(self.i32, 0)
                    insertpt = arith.OrIOp(zero, zero)
                    for op in list(else_block.operations):
                        op.move_after(insertpt)
                        insertpt = op

                return if_op.results

            case rg.Loop(body=rg.RegionEnd() as body, operands=operands):
                raise NotImplementedError
                rettys = Attributes(body.begin.attrs)
                # process operands
                ops = []
                for op in operands:
                    ops.append((yield op))

                result_tys = []
                for i in range(1, rettys.num_output_types() + 1):
                    out_ty = rettys.get_output_type(i)
                    if out_ty is not None:
                        match out_ty.name:
                            case "Int64":
                                result_tys.append(self.i64)
                            case "Float64":
                                result_tys.append(self.f64)
                            case "Bool":
                                result_tys.append(self.boolean)
                    else:
                        result_tys.append(self.i32)

                while_op = scf.WhileOp(
                    results_=result_tys, inits=[op for op in ops]
                )
                before_block = while_op.before.blocks.append(*result_tys)
                after_block = while_op.after.blocks.append(*result_tys)
                new_ops = before_block.arguments

                # Before Region
                with ir.InsertionPoint(before_block), state.push(new_ops):
                    values = yield body
                    scf.ConditionOp(
                        args=[val for val in values[1:]], condition=values[0]
                    )

                # After Region
                with ir.InsertionPoint(after_block):
                    scf.YieldOp(after_block.arguments)

                while_op_res = scf._get_op_results_or_values(while_op)
                return while_op_res

            case _:
                raise NotImplementedError(
                    expr, type(expr), ase.as_tuple(expr, depth=3)
                )

    # ## JIT Compilation
    #
    # Implement JIT compilation for MLIR modules using the MLIR execution
    # engine.

    def jit_compile(self, llmod, func_node: rg.Func, func_name="func"):
        """JIT Compilation

        Convert the MLIR module into a JIT-callable function using the MLIR
        execution engine.
        """
        # attributes = Attributes(func_node.body.begin.attrs)
        # Convert SealIR types into MLIR types
        # with self.loc:
        #     input_types = tuple(
        #         [self.lower_type(x) for x in attributes.input_types()]
        #     )

        # output_types = (
        #     self.lower_type(
        #         Attributes(func_node.body.begin.attrs).get_return_type(
        #             func_node.body
        #         )
        #     ),
        # )
        input_types = ()
        output_types = ()
        from ctypes.util import find_library

        needed_shared_libs = ("mlir_c_runner_utils", "mlir_runner_utils")
        shared_libs = [find_library(x) for x in needed_shared_libs]
        import os.path

        shared_libs.append(os.path.abspath("./libnbrt.so"))
        print(shared_libs)
        return self.jit_compile_extra(llmod, input_types, output_types)

    def jit_compile_extra(
        self,
        llmod,
        input_types,
        output_types,
        function_name="func",
        exec_engine=None,
        is_ufunc=False,
        **execution_engine_params,
    ):
        # Converts the MLIR module into a JIT-callable function.
        # Use MLIR's own internal execution engine
        if exec_engine is None:
            engine = execution_engine.ExecutionEngine(
                llmod, **execution_engine_params
            )
        else:
            engine = exec_engine

        assert len(output_types) in (
            0,
            1,
        ), "Execution of functions with output arguments > 1 not supported"
        nout = len(output_types)

        # Build a wrapper function
        def jit_func(*args):
            if is_ufunc:
                input_args = args[:-nout]
                output_args = args[-nout:]
            else:
                input_args = args
                output_args = [None]
            assert len(input_args) == len(input_types)
            for arg, arg_ty in zip(input_args, input_types):
                # assert isinstance(arg, arg_ty)
                # TODO: Assert types here
                pass

            if False:
                # Transform the input arguments into C-types
                # with their respective values. All inputs to
                # the internal execution engine should
                # be C-Type pointers.
                input_exec_ptrs = [
                    self.get_exec_ptr(ty, val)[0]
                    for ty, val in zip(input_types, input_args)
                ]
                # Invokes the function that we built, internally calls
                # _mlir_ciface_function_name as a void pointer with the given
                # input pointers, there can only be one resulting pointer
                # appended to the end of all input pointers in the invoke call.
                res_ptr, res_val = self.get_exec_ptr(
                    output_types[0], output_args[0]
                )
                engine.invoke(function_name, *input_exec_ptrs, res_ptr)
            else:
                engine.invoke(function_name)
                return

            return self.get_out_val(res_ptr, res_val)

        return jit_func

    @classmethod
    def get_exec_ptr(self, mlir_ty, val):
        """Get Execution Pointer

        Convert MLIR types to C-types and allocate memory for the value.
        """
        if isinstance(mlir_ty, ir.IntegerType):
            val = 0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_int64(val))
        elif isinstance(mlir_ty, ir.F32Type):
            val = 0.0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_float(val))
        elif isinstance(mlir_ty, ir.F64Type):
            val = 0.0 if val is None else val
            ptr = ctypes.pointer(ctypes.c_double(val))
        elif isinstance(mlir_ty, ir.MemRefType):
            if isinstance(mlir_ty.element_type, ir.F64Type):
                np_dtype = np.float64
            elif isinstance(mlir_ty.element_type, ir.F32Type):
                np_dtype = np.float32
            else:
                raise TypeError(
                    "The current array element type is not supported"
                )

            if val is None:
                if not mlir_ty.has_static_shape:
                    raise ValueError(f"{mlir_ty} does not have static shape")
                val = np.zeros(mlir_ty.shape, dtype=np_dtype)

            ptr = ctypes.pointer(
                ctypes.pointer(runtime.get_ranked_memref_descriptor(val))
            )

        return ptr, val

    @classmethod
    def get_out_val(cls, res_ptr, res_val):
        if isinstance(res_val, np.ndarray):
            return res_val
        else:
            return res_ptr.contents.value

    def declare_builtins(self, sym_name, argtypes, restypes):
        if sym_name in self._declared:
            return self._declared[sym_name]

        with self.module_body:
            ret = self._declared[sym_name] = func.FuncOp(
                sym_name,
                (argtypes, restypes),
                visibility="private",
            )
        return ret
