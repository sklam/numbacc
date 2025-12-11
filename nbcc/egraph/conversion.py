from sealir.eqsat.rvsdg_extract_details import EGraphToRVSDG as _EGraphToRVSDG

from ..frontend import grammar as sg


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
            case "CallFQN", {"fqn": fqn, "io": io, "args": args}:
                return grm.write(sg.CallFQN(fqn=fqn, io=io, args=args))
            case "CalleeFQN", {"fullname": str(fullname)}:
                return grm.write(sg.FQN(fullname))
            case "FQN.function", {"fullname": str(fullname)}:
                fqn = grm.write(sg.FQN(fullname))
                return fqn

            case _:
                # Use parent's implementation for other terms.
                return super().handle_Term(op, children, grm)

    def handle_Metadata(
        self, key: str, op: str, children: dict | list, grm: sg.Grammar
    ):
        match op, children:
            case "Metadata.typeinfo", {
                "value": value,
                "type_expr": type_expr,
            }:
                return grm.write(sg.TypeInfo(value=value, type_expr=type_expr))
            case "Metadata.irtag", {
                "value": value,
                "tag": tag,
                "data": data,
            }:
                return grm.write(sg.IRTag(value=value, tag=tag, data=data))
        raise NotImplementedError(key, op, children)

    def handle_TypeExpr(
        self, key: str, op: str, children: dict | list, grm: sg.Grammar
    ):
        match op, children:
            case "TypeExpr.simple", {"name": str(name)}:
                return grm.write(sg.TypeExpr(name=name, args=()))
            case "TypeExpr.function", {"args": args}:
                return grm.write(
                    sg.TypeExpr(name=".function", args=args.children)
                )
        raise NotImplementedError(op, children)

    def handle_IRTagData(
        self, key: str, op: str, children: dict | list, grm: sg.Grammar
    ):
        match op, children:
            case "IRTagData", {"key": str(key), "value": str(value)}:
                return grm.write(sg.IRTagData(key=key, value=value))
        raise NotImplementedError(op, children)
