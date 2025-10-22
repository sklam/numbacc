import egglog
import sealir.eqsat.rvsdg_eqsat as rvsdg
import sealir.eqsat.py_eqsat as py
from egglog import union

Term = rvsdg.Term
TermList = rvsdg.TermList
_w = rvsdg.wildcard


def egraph_optimize(egraph: egglog.EGraph):
    rule_schedule = make_schedule()
    egraph.run(rule_schedule)


def make_schedule() -> egglog.Schedule:
    return (
        ruleset_simplify_builtin_arith | ruleset_simplify_builtin_print
    ).saturate()


@egglog.function
def Op_i32_add(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Builtin_print_i32(io: Term, arg: Term) -> Term: ...


@egglog.function
def Builtin_struct__make__(args: TermList) -> Term: ...


@egglog.function
def Builtin_struct__get_field__(struct: Term, pos: egglog.i64) -> Term: ...


@egglog.ruleset
def ruleset_simplify_builtin_arith(
    io: Term, lhs: Term, rhs: Term, argvec: egglog.Vec[Term], call: Term
):
    yield egglog.rule(
        call
        == py.Py_Call(
            io=io,
            func=py.Py_LoadGlobal(io=_w(Term), name="operator::i32_add"),
            args=rvsdg.TermList(argvec),
        ),
        argvec[0] == lhs,
        argvec[1] == rhs,
        argvec.length() == 2,
    ).then(
        union(call.getPort(0)).with_(io),
        union(call.getPort(1)).with_(Op_i32_add(lhs, rhs)),
    )


@egglog.ruleset
def ruleset_simplify_builtin_print(
    io: Term, printee: Term, argvec: egglog.Vec[Term], call: Term
):
    yield egglog.rule(
        call
        == py.Py_Call(
            io=io,
            func=py.Py_LoadGlobal(io=_w(Term), name="builtins::print_i32"),
            args=rvsdg.TermList(argvec),
        ),
        argvec[0] == printee,
        argvec.length() == 1,
    ).then(
        union(call.getPort(0)).with_(Builtin_print_i32(io, printee)),
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
