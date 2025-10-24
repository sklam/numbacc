import egglog
import sealir.rvsdg.grammar as rg
from sealir.grammar import Grammar as _Grammar, Rule
from sealir.ase import SExpr


class _Root(Rule):
    pass


class BuiltinOp(_Root):
    opname: str
    args: tuple[SExpr]


class TypeInfo(_Root):
    value: SExpr
    typename: str


class Grammar(_Grammar):
    start = rg.Grammar.start | _Root
