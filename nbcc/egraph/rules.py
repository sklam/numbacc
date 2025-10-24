from __future__ import annotations

import warnings

import egglog
import sealir.eqsat.py_eqsat as py
import sealir.eqsat.rvsdg_eqsat as rvsdg
import sealir.rvsdg.grammar as rg
from egglog import union
from sealir.ase import SExpr

from ..frontend import grammar as sg

Term = rvsdg.Term
TermList = rvsdg.TermList
_w = rvsdg.wildcard


def egraph_optimize(egraph: egglog.EGraph):
    rule_schedule = make_schedule()
    egraph.run(rule_schedule)


def make_schedule() -> egglog.Schedule:
    return (
        ruleset_simplify_builtin_arith
        | ruleset_simplify_builtin_print
        | ruleset_typing
    ).saturate()


def egraph_convert_metadata(mdlist: list[SExpr], memo) -> egglog.Vec[Metadata]:
    def gen(md):
        match md:
            case rg.DbgValue(name, value, srcloc, interloc):
                warnings.warn("skip DbgValue")
            case sg.TypeInfo(value=value, typename=str(typename)):
                if value in memo:
                    return Metadata.typeinfo(memo[value], typename)
            case _:
                raise NotImplementedError

    return egglog.Vec[Metadata](
        *filter(lambda x: x is not None, map(gen, mdlist))
    )


class Metadata(egglog.Expr):
    @classmethod
    def typeinfo(
        cls, value: Term, typename: egglog.StringLike
    ) -> Metadata: ...


@egglog.function
def Md_type_info(value: Term, typename: egglog.StringLike) -> Term: ...


@egglog.function
def Op_i32_add(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_sub(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_lt(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_gt(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_not(operand: Term) -> Term: ...


@egglog.function
def Builtin_print_i32(io: Term, arg: Term) -> Term: ...


@egglog.function
def Builtin_print_str(io: Term, arg: Term) -> Term: ...


@egglog.function
def Builtin_struct__make__(args: TermList) -> Term: ...


@egglog.function
def Builtin_struct__get_field__(struct: Term, pos: egglog.i64) -> Term: ...


@egglog.ruleset
def ruleset_simplify_builtin_arith(
    io: Term,
    operand: Term,
    lhs: Term,
    rhs: Term,
    argvec: egglog.Vec[Term],
    call: Term,
):
    BINOPS = {
        "operator::i32_add": Op_i32_add,
        "operator::i32_sub": Op_i32_sub,
        "operator::i32_gt": Op_i32_gt,
        "operator::i32_lt": Op_i32_lt,
    }
    for fname, ctor in BINOPS.items():
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=rvsdg.TermList(argvec),
            ),
            argvec[0] == lhs,
            argvec[1] == rhs,
            argvec.length() == 2,
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(ctor(lhs, rhs)),
        )

    # Handle PyUnaryOp not
    yield egglog.rule(
        call
        == py.Py_NotIO(
            io=io,
            term=operand,
        ),
    ).then(
        union(call.getPort(0)).with_(io),
        union(call.getPort(1)).with_(Op_i32_not(operand)),
    )


@egglog.ruleset
def ruleset_simplify_builtin_print(
    io: Term, printee: Term, argvec: egglog.Vec[Term], call: Term
):
    KNOWN_PRINTS = {
        "builtins::print_i32": Builtin_print_i32,
        "builtins::print_str": Builtin_print_str,
    }

    for fname, builtin_ctor in KNOWN_PRINTS.items():
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=rvsdg.TermList(argvec),
            ),
            argvec[0] == printee,
            argvec.length() == 1,
        ).then(
            union(call.getPort(0)).with_(builtin_ctor(io, printee)),
            union(call.getPort(0)).with_(call.getPort(1)),
        )


def create_ruleset_struct__make__(w_obj):
    fname = w_obj.fqn.fullname

    def ruleset_struct__make__(io: Term, args: TermList, call: Term):
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=args,
            ),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(Builtin_struct__make__(args)),
        )

    ruleset_struct__make__.__name__ += fname

    return egglog.ruleset(ruleset_struct__make__)


def create_ruleset_struct__get_field__(w_obj, field_pos: int):
    fname = w_obj.fqn.fullname

    def ruleset_struct__get_field__(
        io: Term,
        args: TermList,
        call: Term,
        argvec: egglog.Vec[Term],
        struct: Term,
    ):
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=TermList(argvec),
            ),
            argvec[0] == struct,
            argvec.length() == 1,
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(
                Builtin_struct__get_field__(struct, field_pos)
            ),
        )

    ruleset_struct__get_field__.__name__ += fname
    return egglog.ruleset(ruleset_struct__get_field__)


@egglog.ruleset
def ruleset_typing(x: Term):
    if False:
        yield
