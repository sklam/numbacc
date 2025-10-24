import egglog
import sealir.rvsdg.grammar as rg
from sealir.grammar import Grammar as _Grammar, Rule
from sealir.ase import SExpr


class _Root(Rule):
    pass


class BuiltinOp(_Root):
    opname: str
    args: tuple[SExpr]


# class VarAnnotation(_Root):
#     typename: str
#     symbol: str  # for debug
#     value: SExpr


class Grammar(_Grammar):
    start = rg.Grammar.start | _Root
