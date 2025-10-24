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
            case "Op_i32_sub", {"lhs": lhs, "rhs": rhs}:
                return grm.write(
                    sg.BuiltinOp(opname="i32_sub", args=(lhs, rhs))
                )
            case "Op_i32_gt", {"lhs": lhs, "rhs": rhs}:
                return grm.write(
                    sg.BuiltinOp(opname="i32_gt", args=(lhs, rhs))
                )
            case "Op_i32_lt", {"lhs": lhs, "rhs": rhs}:
                return grm.write(
                    sg.BuiltinOp(opname="i32_lt", args=(lhs, rhs))
                )
            case "Op_i32_not", {"operand": operand}:
                return grm.write(
                    sg.BuiltinOp(opname="i32_not", args=(operand,))
                )
            case "Builtin_print_i32", {"io": io, "arg": arg}:
                return grm.write(
                    sg.BuiltinOp(opname="print_i32", args=(io, arg))
                )
            case "Builtin_print_str", {"io": io, "arg": arg}:
                return grm.write(
                    sg.BuiltinOp(opname="print_str", args=(io, arg))
                )
            case "Builtin_struct__make__", {"args": args}:
                return grm.write(
                    sg.BuiltinOp(opname="struct_make", args=tuple(args))
                )
            case "Builtin_struct__get_field__", {"struct": struct, "pos": pos}:
                return grm.write(
                    sg.BuiltinOp(opname="struct_get", args=(struct, pos))
                )
            case _:
                # Use parent's implementation for other terms.
                return super().handle_Term(op, children, grm)
