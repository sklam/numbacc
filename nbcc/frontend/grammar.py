from __future__ import annotations
import sealir.rvsdg.grammar as rg
from sealir.grammar import Grammar as _Grammar, Rule
from sealir.ase import SExpr


class _Root(Rule):
    pass


class BuiltinOp(_Root):
    opname: str
    args: tuple[SExpr, ...]


class IRTag(_Root):
    value: SExpr
    tag: str
    data: tuple[SExpr, ...]


class IRTagData(_Root):
    key: str
    value: str


class TypeExpr(_Root):
    name: str
    args: tuple[SExpr, ...]


class TypeInfo(_Root):
    value: SExpr
    type_expr: SExpr


class FQN(_Root):
    fullname: str


class TypedFQN(_Root):
    fullname: str


class CallFQN(_Root):
    fqn: SExpr
    io: SExpr
    args: tuple[SExpr]


class MLIRInlineAsm(_Root):
    asm: str
    io: SExpr
    args: tuple[SExpr]


class Grammar(_Grammar):
    start = rg.Grammar.start | _Root
