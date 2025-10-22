from typing import Sequence
from contextlib import contextmanager
from pprint import pprint
from collections import defaultdict
from dataclasses import dataclass, field

from spy.fqn import FQN
from spy.interop import redshift
from spy.vm.function import W_ASTFunc

from numba_scfg.core.datastructures.basic_block import (
    RegionBlock,
    BasicBlock,
    SyntheticAssignment,
    SyntheticTail,
    SyntheticHead,
    SyntheticFill,
    SyntheticReturn,
)
from numba_scfg.core.datastructures.scfg import SCFG
from .restructure import restructure, _SpyScfgRenderer, SCFG, SpyBasicBlock
from .spy_ast import Node, convert_to_node

from sealir import ase
from sealir.rvsdg import grammar as rg
from sealir.rvsdg import internal_prefix, format_rvsdg


class TranslationUnit:
    def __init__(self):
        self._symtabs = {}

    def add(self, fqn: FQN, region: SCFG) -> None:
        self._symtabs[fqn] = region

    def get_function(self, name: str) -> tuple[FQN, Node]:
        for fqn in self._symtabs:
            if fqn.symbol_name == name:
                return fqn, self._symtabs[fqn]
        raise NameError(name)

    def __repr__(self):
        cname = self.__class__.__name__
        syms = ", ".join(map(str, self._symtabs))
        return f"{cname}([{syms}])"


def frontend(filename: str, *, view: bool = False) -> TranslationUnit:
    vm, w_mod = redshift(filename)

    symtab: dict[FQN, Node] = {}
    for fqn, w_obj in vm.fqns_by_modname(w_mod.name):
        print(fqn, w_obj)
        if isinstance(w_obj, W_ASTFunc):
            print("functype:", w_obj.w_functype)
            if w_obj.locals_types_w is not None:
                node = convert_to_node(w_obj.funcdef, vm=vm)
                pprint(node)
                symtab[fqn] = node
                print()

    # restructure
    tu = TranslationUnit()
    for fqn, func_node in symtab.items():

        scfg = restructure(fqn.fullname, func_node)
        if view:
            _SpyScfgRenderer(scfg).view()
        region = convert_to_sexpr(func_node, scfg)
        print(format_rvsdg(region))
        tu.add(fqn, region)

    return tu


def convert_to_sexpr(func_node: Node, scfg: SCFG):
    with ase.Tape() as tape:
        cts = ConvertToSExpr(tape)
        with cts.setup_function(func_node) as rb:
            cts.handle_region(scfg)

        region = cts.close_function(rb, func_node)
        return region


@dataclass(frozen=True)
class Scope:
    vardefs: dict[str, FQN] = field(init=False, default_factory=dict)
    local_vars: dict[str, ase.SExpr] = field(init=False, default_factory=dict)


@dataclass(frozen=True)
class ConversionContext:
    grm: rg.Grammar
    scope_stack: list = field(init=False, default_factory=list)
    scope_map: dict[rg.RegionBegin, Scope] = field(
        init=False, default_factory=dict
    )

    @property
    def scope(self) -> Scope:
        return self.scope_stack[-1]

    def store_local(self, target: str, expr: ase.SExpr) -> None:
        self.scope.local_vars[target] = expr

    def load_local(self, target: str) -> ase.SExpr:
        return self.scope.local_vars[target]

    def get_io(self) -> ase.SExpr:
        return self.load_local(internal_prefix("io"))

    def set_io(self, value: ase.SExpr) -> None:
        self.store_local(internal_prefix("io"), value)

    def insert_io_node(self, node: ase.SExpr) -> ase.SExpr:
        grm = self.grm
        written = grm.write(node)
        io, res = (grm.write(rg.Unpack(val=written, idx=i)) for i in range(2))
        self.set_io(io)
        return res

    def update_scope(self, expr: ase.SExpr, vars: Sequence[str]):
        grm = self.grm

        for i, k in enumerate(vars):
            self.store_local(k, grm.write(rg.Unpack(val=expr, idx=i)))

    @contextmanager
    def new_region(self, region_parameters: Sequence[str]):
        write = self.grm.write
        rb = write(
            rg.RegionBegin(
                attrs=write(rg.Attrs(())), inports=tuple(region_parameters)
            )
        )

        scope = Scope()
        self.scope_map[rb] = scope
        self.scope_stack.append(scope)

        self.initialize_scope(rb)

        yield rb

        self.scope_stack.pop()

    def initialize_scope(self, rb: rg.RegionBegin):
        write = self.grm.write
        for i, k in enumerate(rb.inports):
            self.store_local(k, write(rg.Unpack(val=rb, idx=i)))

    def compute_updated_vars(self, rb: rg.RegionBegin) -> set[str]:
        return set(self.scope_map[rb].local_vars.keys())

    def close_region(
        self, rb: rg.RegionBegin, expected_vars: set[str]
    ) -> rg.RegionEnd:
        scope = self.scope_map[rb]

        write = self.grm.write
        ports: list[rg.Port] = []
        for k in sorted(expected_vars):
            if k not in scope.local_vars:
                v = write(rg.Undef(name=k))
            else:
                v = scope.local_vars[k]
            p = rg.Port(name=k, value=v)
            ports.append(write(p))

        return write(rg.RegionEnd(begin=rb, ports=tuple(ports)))

    def get_scope_as_operands(self) -> tuple[ase.SExpr, ...]:
        operands = []
        for _, v in sorted(self.scope.local_vars.items()):
            operands.append(v)
        return tuple(operands)

    def get_scope_as_parameters(self) -> tuple[str, ...]:
        return tuple(sorted(self.scope.local_vars))


class ConvertToSExpr:
    def __init__(self, tape: ase.Tape):
        self._tape = tape
        self._context = ConversionContext(grm=rg.Grammar(self._tape))

    @contextmanager
    def setup_function(self, func_node: Node):
        match func_node:
            case Node("FuncDef", args=args):
                if args:
                    raise NotImplementedError("arguments handling")
            case _:
                raise ValueError(func_node)

        ctx = self._context

        with ctx.new_region([internal_prefix("io")]) as rb:
            yield rb

    def close_function(self, rb: rg.RegionBegin, func_node: Node) -> rg.Func:
        ctx = self._context
        vars = ctx.compute_updated_vars(rb)

        name = func_node.name
        assert not func_node.args
        args = ctx.grm.write(rg.Args(()))
        return ctx.grm.write(
            rg.Func(fname=name, args=args, body=ctx.close_region(rb, vars))
        )

    def handle_region(self, scfg: SCFG):
        ctx = self._context
        crv = list(scfg.concealed_region_view.items())
        by_kinds = defaultdict(list)
        for _, block in crv:
            kind = getattr(block, "kind", None)
            by_kinds[kind].append(block)

        print("--by-kinds", [(k, len(vs)) for k, vs in by_kinds.items()])
        if "branch" in by_kinds:
            [head_block] = by_kinds["head"]
            [then_block, else_block] = by_kinds["branch"]
            [tail_block] = by_kinds["tail"]

            test_expr = self.codegen(head_block)

            operands = ctx.get_scope_as_operands()

            with ctx.new_region(ctx.get_scope_as_parameters()) as rb_then:
                self.codegen(then_block)

            with ctx.new_region(ctx.get_scope_as_parameters()) as rb_else:
                self.codegen(else_block)

            updated_vars = ctx.compute_updated_vars(rb_then)
            updated_vars |= ctx.compute_updated_vars(rb_else)

            region_then = ctx.close_region(rb_then, updated_vars)
            region_else = ctx.close_region(rb_else, updated_vars)

            ifelse = ctx.grm.write(
                rg.IfElse(
                    cond=test_expr,
                    body=region_then,
                    orelse=region_else,
                    operands=operands,
                ),
            )
            ctx.update_scope(ifelse, sorted(updated_vars))
            return self.codegen(tail_block)

        elif len(by_kinds) == 1 and None in by_kinds:
            for blk in by_kinds[None]:
                last = self.codegen(blk)
            return last
        else:
            assert False, "not branch"

    def codegen(self, block: BasicBlock) -> ase.SExpr:
        print("AT", block.name)
        ctx = self._context
        grm = ctx.grm
        match block:
            case RegionBlock():
                if isinstance(block.subregion, SCFG):
                    return self.handle_region(block.subregion)
                else:
                    return self.codegen(block.subregion)

            case SpyBasicBlock():
                if not block.body:
                    return
                assert len(block.body) > 0
                last_expr: ase.Expr
                for stmt in block.body:
                    last_expr = self.emit_statement(stmt)
                return last_expr

            case SyntheticAssignment():
                for k, v in block.variable_assignment.items():
                    match v:
                        case int(ival):
                            const = grm.write(rg.PyInt(ival))
                        case _:
                            raise ValueError(type(v))
                    ctx.store_local(k, const)

            case SyntheticReturn():
                return ctx.load_local("__scfg_return_value__")
            case SyntheticTail() | SyntheticHead() | SyntheticFill():
                # These are empty blocks
                return ctx.get_io()
            case _:
                raise AssertionError(type(block))

    def emit_statement(self, stmt: Node) -> ase.SExpr:
        ctx = self._context
        grm = ctx.grm
        match stmt:
            case Node(
                "VarDef",
                kind="var",
                name=str(name),
                type=Node(
                    "FQNConst", fqn=Node("literal", value=FQN() as type_fqn)
                ),
            ):
                ctx.scope.vardefs[name] = type_fqn
                return
            case Node(
                "AssignLocal",
                target=Node("StrConst", value=str(target)),
                value=rval,
            ):
                expr = self.emit_expression(rval)
                ctx.store_local(target, expr)
                return expr

            case Node("StmtExpr", value=Node() as value):
                last = self.emit_expression(value)
                ctx.store_local(internal_prefix("last"), last)
                return last
            case Node("Call"):
                return self.emit_expression(stmt)
            case Node("Return"):
                ret = self.emit_expression(stmt.value)
                ctx.store_local("__scfg_return_value__", ret)
                return ret
            case _:
                raise NotImplementedError(stmt)

    def emit_expression(self, node: Node) -> ase.SExpr:
        ctx = self._context
        grm = ctx.grm
        match node:
            case Node("NameLocal"):
                return ctx.load_local(node.sym.name)
            case Node(
                "Call",
                func=Node(
                    "FQNConst", fqn=Node("literal", value=FQN() as callee_fqn)
                ),
                args=list(args),
            ):
                callee = grm.write(
                    rg.PyLoadGlobal(io=ctx.get_io(), name=str(callee_fqn))
                )

                return ctx.insert_io_node(
                    rg.PyCall(
                        io=ctx.get_io(),
                        func=callee,
                        args=tuple(map(self.emit_expression, args)),
                    )
                )
            case Node("Constant", value=int(ival)):
                return grm.write(rg.PyInt(ival))

            case Node("Constant", value=None):
                return grm.write(rg.PyNone())

            case Node("NameLocal"):
                return ctx.load_local(node.sym.name)
            case _:
                raise NotImplementedError(node)
