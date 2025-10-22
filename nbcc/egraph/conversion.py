from sealir.eqsat.rvsdg_extract_details import EGraphToRVSDG as _EGraphToRVSDG
from . import grammar as sg


class ExtendEGraphToRVSDG(_EGraphToRVSDG):
    grammar = sg.Grammar

    def handle_Term(self, op: str, children: dict | list, grm: sg.Grammar):
        match op, children:
            case "Op_i32_add", {"lhs": lhs, "rhs": rhs}:
                return grm.write(
                    sg.BuiltinOp(opname="i32_add", args=(lhs, rhs))
                )
            case "Builtin_print_i32", {"io": io, "arg": arg}:
                return grm.write(
                    sg.BuiltinOp(opname="print_i32", args=(io, arg))
                )
            case _:
                # Use parent's implementation for other terms.
                return super().handle_Term(op, children, grm)
