from __future__ import annotations

import base64
from collections import defaultdict
import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Sequence

import mlir.dialects.arith as arith
import mlir.dialects.cf as cf
import mlir.dialects.func as func
import mlir.dialects.scf as scf
import mlir.execution_engine as execution_engine
import mlir.ir as ir

import mlir.runtime as runtime
from mlir.dialects.transform.interpreter import apply_named_sequence

from mlir.ir import _GlobalDebug


from spy.fqn import FQN

from .mlir_passes import PassManager
import numpy as np

from mlir.dialects import llvm
from sealir import ase
from sealir.rvsdg import grammar as rg
from sealir.rvsdg import internal_prefix

from nbcc.developer import TODO
from ..frontend import grammar as sg

# ## MLIR Backend Implementation
#
# Define the core MLIR backend class that handles type lowering and
# expression compilation.


# _GlobalDebug.flag = True
_DEBUG = True


@dataclass(frozen=True)
class LowerStates(ase.TraverseState):
    push: Callable
    get_region_args: Callable
    function_block: func.FuncOp
    constant_block: ir.Block


class MDMap:
    mdmap: defaultdict[ase.SExpr, list[ase.SExpr]]

    def __init__(self):
        self.mdmap = defaultdict(list)

    def load(self, mdlist):
        for md in mdlist:
            match md:
                case sg.TypeInfo(value=value) | sg.IRTag(value=value):
                    self.mdmap[value].append(md)
                case _:
                    breakpoint()
                    TODO(f"Unknown MD: {md}: {type(md)}")

    def lookup_irtag(self, val: ase.SExpr) -> list[sg.TypeInfo]:
        return [x for x in self.mdmap[val] if isinstance(x, sg.IRTag)]

    def lookup_typeinfo(self, val: ase.SExpr) -> list[sg.TypeInfo]:
        return [x for x in self.mdmap[val] if isinstance(x, sg.TypeInfo)]

    def lookup_typeinfo_by_fqn(self, fqn: str) -> list[sg.TypeInfo]:
        for md, tis in self.mdmap.items():
            match md:
                case sg.FQN(fullname=fqn_fullname):
                    if fqn_fullname == fqn:
                        return [x for x in tis if isinstance(x, sg.TypeInfo)]
        raise NameError(f"{fqn!r} not found")


class Backend:

    def __init__(self):
        self.context = context = ir.Context()
        context.enable_multithreading(False)
        # context.allow_unregistered_dialects = True
        with context, ir.Location.name("Backend.__init__"):
            self.f32 = ir.F32Type.get(context=context)
            self.f64 = ir.F64Type.get(context=context)
            self.index_type = ir.IndexType.get(context=context)
            self.i8 = ir.IntegerType.get_signless(8, context=context)
            self.i32 = ir.IntegerType.get_signless(32, context=context)
            self.i64 = ir.IntegerType.get_signless(64, context=context)
            self.boolean = ir.IntegerType.get_signless(1, context=context)
            self.io_type = ir.IntegerType.get_signless(1, context=context)
            self.llvm_ptr = ir.Type.parse("!llvm.ptr")
            self.none_type = ir.Type.parse("!llvm.struct<()>")

            self.unranked_tensor_f64 = ir.UnrankedTensorType.get(self.f64)
            self.unranked_memref_f64 = ir.UnrankedMemRefType.get(
                self.f64, memory_space=None
            )
            unknown_dim = ir.ShapedType.get_dynamic_size()
            self.tensor_1d_f64 = ir.RankedTensorType.get(
                shape=[unknown_dim], element_type=self.f64
            )
            self.tensor_2d_f64 = ir.RankedTensorType.get(
                shape=[unknown_dim, unknown_dim], element_type=self.f64
            )
            self.memref_1d_f64 = ir.MemRefType.get(
                shape=[unknown_dim], element_type=self.f64
            )

    def lower_type(self, ty: sg.TypeExpr) -> ir.Type:
        """Type Lowering

        Convert SealIR types to MLIR types for compilation.
        """
        match ty:
            case sg.TypeExpr(name=str(name), args=()):
                fqn = FQN(name)
                if name == "mlir::type::()":
                    return None
                elif fqn.namespace.fullname == "mlir::type":
                    return ir.Type.parse(fqn.symbol_name, context=self.context)
                else:
                    match name:
                        case "builtins::i32":
                            return self.i32
                        case "types::NoneType":
                            return self.none_type

                        case "mlir_tensor_lib::make_tensor_type[mlir::type::f64]::TensorType":
                            TODO(
                                "TODO: lower_type mlir_tensor_lib::make_tensor_type[f64]::TensorType "
                            )
                            return self.tensor_1d_f64
                        case "llm_tensor::make_tensor_type_2d[mlir::type::f64]::TensorType":
                            TODO(
                                "TODO: lower_type mlir_tensor_lib::make_tensor_type[f64]::TensorType "
                            )
                            return self.tensor_2d_f64

        raise NotImplementedError(f"unknown type: {ty}")

    def get_ll_type(self, expr: ase.SExpr, mdmap: MDMap) -> sg.TypeInfo | None:
        mds = mdmap.lookup_typeinfo(expr)
        if not mds:
            return None
        [ty] = mds
        return self.lower_type(ty.type_expr)

    def make_module(self, module_name: str) -> ir.Module:
        with self.context:
            return ir.Module.create(loc=ir.Location.name(module_name))

    def _make_pass_pipeline(self, *passes, with_subprocess=True):
        return PassManager(passes, with_subprocess=with_subprocess)

    def run_passes(
        self, module: ir.Module, transforms: dict[str, Sequence[str]]
    ) -> ir.Module:
        """MLIR Pass Pipeline

        Apply MLIR passes for optimization and lowering to LLVM IR.
        """
        from . import mlir_passes as mp

        for name in transforms:
            self._add_noinline_to_callsite(module, name)

        module = self._make_pass_pipeline(
            mp.Canonicalize(),
            mp.Inline(),
        ).run(module.operation)

        print("After Phase 1")

        for fname, pass_seq in transforms.items():
            self._run_per_function_transform(module, fname, pass_seq)

        module = self._make_pass_pipeline(
            mp.Canonicalize(),
            mp.LinalgGeneralizeNamedOps(),
            mp.LinalgFuseElementwiseOps(),
            mp.LoopInvariantCodeMotion(),
            mp.FoldMemRefAliasOps(),
            # Vector
            mp.LowerVectorMask(),
            mp.Canonicalize(),
            mp.FoldTensorSubsetOps(),  # folds tensor-slice into vector-transfer
            mp.Canonicalize(),
        ).run(module.operation)
        print("After Phase 3 (cleanup)")

        module = self._make_pass_pipeline(
            mp.EliminateEmptyTensors(),
            mp.EmptyTensorToAllocTensor(),
            # Bufferization
            mp.OneShotBufferize(bufferize_function_boundaries=True),
            mp.ConvertVectorToSCF(),
            mp.Canonicalize(),
            mp.CSE(),
        ).run(module.operation)

        print("After Phase 4 (bufferize)")

        module = self._make_pass_pipeline(
            # Affine passes goes after Bufferize
            mp.ConvertLinalgToAffineLoops(),
            mp.NormalizeMemRefs(),
            mp.Canonicalize(),
            mp.CSE(),
            mp.SymbolDCE(),
            mp.ExpandStridedMetadata(),
            mp.AffineScalrep(),
            mp.AffineSimplifyStructures(),
            mp.AffineLoopFusion(mode="greedy", maximal=1),
            # mp.ConvertLinalgToParallelLoops(),
            mp.LowerAffine(),
            mp.Canonicalize(),
            mp.PromoteBuffersToStack(),
            mp.BufferHoisting(),
            mp.BufferLoopHoisting(),
            mp.FoldMemRefAliasOps(),
            # The CSE and canonicalize take care of the reminding redundant memref ops
            mp.CSE(),
            mp.Canonicalize(),
        ).run(module.operation)

        print("After Phase 4.1 (SCF ops)")

        module = self._make_pass_pipeline(
            mp.ScfForLoopCanonicalization(),
            mp.ScfForLoopRangeFolding(),
            mp.ScfForLoopToParallel(),
            mp.ScfParallelLoopFusion(),
            mp.Canonicalize(),
        ).run(module.operation)

        print("After Phase 5 (prelower)")

        module = self._make_pass_pipeline(
            mp.OwnershipBasedBufferDeallocation(),
            mp.BufferDeallocationSimplification(),
            mp.BufferizationLowerDeallocations(),
            mp.ConvertBufferizationToMemRef(),
            # Lowering
            mp.Canonicalize(),
            mp.ConvertSCFToCF(),
            mp.ConvertVectorToLLVM(enable_arm_neon=True),
            mp.FinalizeMemRefToLLVM(),
            mp.ConvertMathToLibM(),
            mp.ConvertFuncToLLVM(),
            mp.ConvertIndexToLLVM(),
            mp.ConvertArithToLLVM(),
            mp.ConvertCFToLLVM(),
            mp.ReconileUnrealizedCasts(),
        ).run(module.operation)
        return module

    def _add_noinline_to_callsite(self, module: ir.Module, fname: str):
        def iterate_funcop(module: ir.Module):
            for blk in module.body.region.blocks:
                for modop in blk.operations:
                    if isinstance(modop, func.FuncOp):
                        yield modop

        for f_op in iterate_funcop(module):
            if f_op.name.value == fname:
                attrmap = f_op.attributes
                attrmap["no_inline"] = ir.UnitAttr.get(context=module.context)

    def _run_per_function_transform(
        self, module: ir.Module, fname: str, pass_seq: Sequence[str]
    ):

        def load_transform(tmod: ir.Module):
            [transform_op] = tmod.body.operations

            def runner(payload):
                return apply_named_sequence(payload, transform_op, tmod)

            return runner

        from . import transforms

        for pass_name in pass_seq:
            tmod = ir.Module.parse(
                getattr(transforms, pass_name), context=self.context
            )
            transform = load_transform(tmod)

            for op in module.body.region.blocks[0].operations:
                if op.name.value == fname:
                    fn_op = op
                    break

            transform(fn_op.operation)

            print(f"Transformed: {fname} after {pass_name}")
            print(fn_op.operation.get_asm())


class Lowering:
    be: Backend
    module: ir.Module
    mdmap: MDMap
    loc: ir.Location

    def __init__(
        self,
        be: Backend,
        module: ir.Module,
        mdmap: MDMap,
        func_map: dict[str, ase.SExpr],
    ):
        self.be = be
        self.module = module
        self.mdmap = mdmap
        self.func_map = func_map
        self._declared = {}

    def get_return_types(self, root) -> list[ir.Type]:
        [retval] = [
            port.value
            for port in root.body.ports
            if port.name == internal_prefix("ret")
        ]
        [ti] = self.mdmap.lookup_typeinfo(retval)
        outs = [self.be.lower_type(ti.type_expr)]
        # Remove return None
        if outs == [self.be.none_type]:
            return []
        return outs

    def irtags(self, root: rg.Func) -> dict:
        out = {}
        if irtags := self.mdmap.lookup_irtag(root.body):
            for irtag in irtags:
                bin = out.setdefault(irtag.tag, [])
                for irtagdata in irtag.data[0].children:
                    bin.append((irtagdata.key, irtagdata.value))
        return out

    def lower(self, root: rg.Func) -> func.FuncOp:
        """Expression Lowering

        Lower RVSDG expressions to MLIR operations, handling control flow
        and data flow constructs.
        """
        context = self.be.context
        self.loc = loc = ir.Location.name(f"{self}.lower()", context=context)
        module = self.module

        function_name = root.fname
        # Get the module body pointer so we can insert content into the
        # module.
        self.module_body = module_body = ir.InsertionPoint(module.body)
        # Convert SealIR types to MLIR types.

        input_types = tuple(
            [self.be.lower_type(x) for x in root.args.arguments]
        )
        output_types = self.get_return_types(root)

        with context, loc, module_body:
            # Constuct a function that emits a callable C-interface.
            fnty = func.FunctionType.get(input_types, output_types)
            fqn = FQN(function_name)
            if fqn.symbol_name == "main":
                TODO("XXX: hack main() function handling")
                export_name = "main"
            else:
                export_name = fqn.c_name
            TODO("TODO: is exporting logic")
            is_exporting = export_name == "main" or fqn.symbol_name.startswith(
                "export_"
            )
            fun = func.FuncOp(
                export_name,
                fnty,
                visibility=("public" if is_exporting else "private"),
            )
            if is_exporting:
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

        fun.operation.verify()
        return fun

    def _cast_return_value(self, val):
        return val

    def lower_expr(self, expr: SExpr, state: LowerStates):
        """Expression Lowering Implementation

        Implement the core expression lowering logic for various RVSDG
        constructs including functions, regions, control flow, and operations.
        """

        module = self.module
        context = self.be.context
        match expr:
            case rg.Func(args=args, body=body, fname=fqn):
                TODO("XXX: no way to get return type")
                # [fqn_ti] = self.mdmap.lookup_typeinfo_by_fqn(fqn)
                # resty = fqn_ti.type_expr.args[0]
                # print(fqn)
                # print(resty.name)
                names = {
                    argspec.name: state.function_block.arguments[i]
                    for i, argspec in enumerate(args.arguments)
                }
                argvalues = []
                for k in body.begin.inports:
                    if k == internal_prefix("io"):
                        v = arith.constant(self.be.io_type, 0)
                    else:
                        v = names[k]
                    argvalues.append(v)

                with state.push(argvalues):
                    outs = yield body

                portnames = [p.name for p in body.ports]

                if self.get_return_types(expr) == []:
                    func.ReturnOp([])
                    return
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
                    const = arith.constant(
                        self.be.i32, ival
                    )  # HACK: select type
                return const

            case rg.PyBool(int(ival)):
                with state.constant_block:
                    const = arith.constant(self.be.boolean, ival)
                return const

            case rg.PyFloat(float(fval)):
                with state.constant_block:
                    const = arith.constant(self.be.f64, fval)
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
                            ir.IntegerAttr.get(self.be.i64, length),
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
                    ptr_type = self.be.llvm_ptr
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

            case sg.BuiltinOp("i32_lt", (lhs, rhs)):
                lhs = yield lhs
                rhs = yield rhs
                return arith.cmpi(arith.CmpIPredicate.slt, lhs, rhs)

            case sg.BuiltinOp("i32_gt", (lhs, rhs)):
                lhs = yield lhs
                rhs = yield rhs
                return arith.cmpi(arith.CmpIPredicate.sgt, lhs, rhs)

            case sg.BuiltinOp("i32_not", (operand,)):
                operand = yield operand
                return arith.cmpi(
                    arith.CmpIPredicate.eq,
                    operand,
                    arith.constant(self.be.i32, 0),
                )

            case sg.BuiltinOp("print_i32", (io, operand)):
                io = yield io
                operand = yield operand

                print_fn = self.declare_builtins(
                    "spy_builtins$print_i32", [self.be.i32], []
                )
                func.call(
                    print_fn.type.results, "spy_builtins$print_i32", [operand]
                )
                return io

            case sg.BuiltinOp("print_str", (io, operand)):
                io = yield io
                operand = yield operand

                print_fn = self.declare_builtins(
                    "spy_builtins$print_str", [self.be.llvm_ptr], []
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

                resty = self.be.i32  # HACK
                return llvm.extractvalue(
                    resty, struct_value, ir.DenseI64ArrayAttr.get([pos])
                )

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

                # determine result types
                for left_port, right_port in zip(
                    body.ports, orelse.ports, strict=True
                ):
                    left_ty = self.get_port_type(left_port)
                    right_ty = self.get_port_type(right_port)
                    if left_ty is None:
                        ty = right_ty
                    elif right_ty is None:
                        ty = left_ty
                    else:
                        assert left_ty == right_ty, f"{left_ty} != {right_ty}"
                        ty = left_ty
                    result_tys.append(ty)

                # Build the MLIR If-else
                if_op = scf.IfOp(
                    cond=condval, results_=result_tys, hasElse=True
                )

                with state.push(operand_vals):
                    # Make a detached module to temporarily house the blocks
                    with ir.InsertionPoint(if_op.then_block):
                        value_body = yield body
                        scf.YieldOp([x for x in value_body])

                    with ir.InsertionPoint(if_op.else_block):
                        value_else = yield orelse
                        scf.YieldOp([x for x in value_else])

                return if_op.results

            case rg.Loop(body=rg.RegionEnd() as body, operands=operands):
                # process operands
                operand_vals = []
                for op in operands:
                    operand_vals.append((yield op))

                result_tys = []
                for op in operand_vals:
                    result_tys.append(op.type)

                while_op = scf.WhileOp(
                    results_=result_tys, inits=[op for op in operand_vals]
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

            case rg.Undef(name):
                # HACK
                return arith.constant(self.be.i32, 0)

            case sg.CallFQN(
                fqn=sg.FQN() as callee_fqn, io=io_val, args=args_vals
            ):

                if callee_fqn.fullname.endswith(
                    "::__lift__"
                ) or callee_fqn.fullname.endswith("::__unlift__"):
                    TODO("XXX: lift/unlift lowering is a hack")
                    [val] = args_vals
                    return [(yield io_val), (yield val)]

                mdmap = self.mdmap

                [callee_ti] = mdmap.lookup_typeinfo(callee_fqn)

                resty = self.be.lower_type(callee_ti.type_expr.args[0])
                argtys = []
                for arg in args_vals:
                    [ti] = mdmap.lookup_typeinfo(arg)
                    argtys.append(self.be.lower_type(ti.type_expr))

                lowered_args = []
                for arg in args_vals:
                    lowered_args.append((yield arg))

                c_name = FQN(callee_fqn.fullname).c_name

                fqn = FQN(callee_fqn.fullname)

                if fqn.namespace.fullname == "mlir::op":
                    TODO("XXX: hardcode support of MLIR::OP ")

                    res = self._handle_mlir_op(
                        fqn.symbol_name,
                        resty,
                        lowered_args,
                    )
                    if op := getattr(res, "owner", None):
                        assert op.verify()
                    return [io_val, res]
                    # self.declare_builtins(c_name, argtys, [resty])
                elif fqn.namespace.fullname == "mlir::asm":
                    res = self._handle_mlir_asm(
                        fqn.symbol_name,
                        resty,
                        lowered_args,
                    )

                    return [io_val, res]
                # if callee_fqn.fullname == "builtins::print_object":
                #     TODO("XXX: hardcode support of builtins::print_object ")
                #     with self.module_body:
                #         self.declare_builtins(c_name, argtys, [resty])
                #         fntype = ir.FunctionType.get(argtys, [resty])
                #         func.FuncOp(
                #             name=c_name,
                #             type=fntype,
                #             visibility="private",
                #         )

                call = func.call([resty], c_name, lowered_args)
                return [io_val, call]
            case rg.PyNone():
                return llvm.mlir_zero(self.be.none_type)
            case _:
                raise NotImplementedError(
                    expr, type(expr), ase.as_tuple(expr, depth=3)
                )

    def _handle_mlir_op(self, mlir_op: str, resty, args):
        from mlir.dialects import linalg, tensor, bufferization

        match mlir_op:
            case "tensor.add":
                [lhs, rhs] = args
                index = arith.constant(self.be.index_type, 0)
                dim = tensor.dim(args[0], index)
                out = tensor.empty([dim], element_type=self.be.f64)
                return linalg.add(lhs, rhs, outs=[out])
            case "linalg.add":
                # linalg.add needs a region
                [lhs, rhs, res] = args
                res = linalg.add(lhs, rhs, outs=[res])
                return res
            case "linalg.sub":
                # linalg.sub needs a region
                [lhs, rhs, res] = args
                res = linalg.sub(lhs, rhs, outs=[res])
                return res
            case "linalg.mul":
                # linalg.mul needs a region
                [lhs, rhs, res] = args
                res = linalg.mul(lhs, rhs, outs=[res])
                return res
            case "linalg.div":
                # linalg.div needs a region
                [lhs, rhs, res] = args
                res = linalg.div(lhs, rhs, outs=[res])
                return res
            case "linalg.exp":
                # linalg.exp needs a region
                [src, res] = args
                res = linalg.exp(src, outs=[res])
                return res
            case "mlir_linalg_reduce_sum_inner_keepdims":
                [arg] = args
                dtype = arg.type.element_type
                c0 = arith.constant(self.be.index_type, 0)
                dim = tensor.dim(arg, c0)
                init = tensor.empty(sizes=[dim], element_type=dtype)
                neg_inf = arith.constant(self.be.f64, float(0))
                init_filled = linalg.fill(neg_inf, outs=[init])
                max_reduce = linalg.reduce(
                    result=[init.type],
                    inputs=[arg],
                    inits=[init_filled],
                    dimensions=[1],
                )

                body = max_reduce.owner.regions[0].blocks.append(dtype, dtype)
                with ir.InsertionPoint(body):
                    linalg.YieldOp(
                        [arith.addf(body.arguments[0], body.arguments[1])]
                    )

                assert max_reduce.owner.verify()

                c1 = arith.constant(self.be.index_type, 1)
                dim1 = tensor.dim(arg, c1)
                output = tensor.empty(sizes=(dim, dim1), element_type=dtype)
                assert output.owner.verify()

                bc = linalg.broadcast(
                    input=max_reduce, outs=[output], dimensions=[1]
                )
                assert bc.verify()
                return bc
            case "mlir_linalg_reduce_max_inner_keepdims":
                [arg] = args
                dtype = arg.type.element_type
                c0 = arith.constant(self.be.index_type, 0)
                dim = tensor.dim(arg, c0)
                init = tensor.empty(sizes=[dim], element_type=dtype)
                neg_inf = arith.constant(self.be.f64, float("-inf"))
                init_filled = linalg.fill(neg_inf, outs=[init])
                max_reduce = linalg.reduce(
                    result=[init.type],
                    inputs=[arg],
                    inits=[init_filled],
                    dimensions=[1],
                )

                body = max_reduce.owner.regions[0].blocks.append(dtype, dtype)
                with ir.InsertionPoint(body):
                    linalg.YieldOp(
                        [arith.maximumf(body.arguments[0], body.arguments[1])]
                    )

                assert max_reduce.owner.verify()

                c1 = arith.constant(self.be.index_type, 1)
                dim1 = tensor.dim(arg, c1)
                output = tensor.empty(sizes=(dim, dim1), element_type=dtype)
                assert output.owner.verify()

                bc = linalg.broadcast(
                    input=max_reduce, outs=[output], dimensions=[1]
                )
                assert bc.verify()
                return bc

                # DYN = ir.ShapedType.get_dynamic_size()
                # expanded = tensor.expand_shape(
                #     result=ir.RankedTensorType.get(
                #         element_type=dtype, shape=[DYN, 1]
                #     ),
                #     src=max_reduce,
                #     reassociation=[[0, 1]],
                #     output_shape=[dim],
                #     static_output_shape=[DYN, 1],
                # )
                # assert expanded.owner.verify()

                # return tensor.cast(
                #     ir.RankedTensorType.get(
                #         element_type=dtype, shape=(DYN, DYN)
                #     ),
                #     expanded,
                # )
            case _:
                raise NotImplementedError(f"Unhandled MLIR op {mlir_op!r}")

    def _handle_mlir_asm(self, mlir_op: str, resty, args):
        mlir_op = base64.urlsafe_b64decode(mlir_op.encode()).decode()
        try:
            first_split = mlir_op.index("$")
        except ValueError:
            pass
        else:
            mlir_op = mlir_op[:first_split]

        opname, _, attr = mlir_op.partition(" ")
        if attr:
            irattrs = ir.Attribute.parse(attr)
            if isinstance(irattrs, ir.DictAttr):
                attrs = {
                    named_attr.name: named_attr.attr for named_attr in irattrs
                }
            else:
                raise ValueError("expects a dictattr")

        else:
            attrs = None

        result_types = [resty] if resty else []
        op = ir.Operation.create(opname, result_types, args, attributes=attrs)
        op.verify()
        if resty:
            return op.result

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

    def get_port_type(self, port) -> ir.Attribute:
        if port.name == internal_prefix("io"):
            ty = self.be.io_type
        else:
            ty = self.be.get_ll_type(port.value, self.mdmap)
        return ty

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
