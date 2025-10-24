import logging
import subprocess as subp
import sys
import tempfile
from contextlib import ExitStack

import sealir.rvsdg.grammar as rg
from egglog import EGraph
from mlir import ir
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import egraph_extraction
from sealir.rvsdg import format_rvsdg

from nbcc.egraph.conversion import ExtendEGraphToRVSDG
from nbcc.egraph.rules import egraph_optimize
from nbcc.frontend import TranslationUnit, frontend
from nbcc.mlir_backend.backend import Backend

logging.disable(logging.INFO)


def compile(path: str, out_path: str) -> None:
    tu = frontend(path)

    rvsdg_ir = middle_end(tu, "main")
    be = Backend()
    module = be.lower(rvsdg_ir, ())
    print(module)
    module.operation.verify()

    module = be.run_passes(module)
    print("After optimization")
    print(module)

    make_binary(module, out_path)


def make_binary(module: ir.Module, out_path: str):
    with ExitStack() as raii:
        temp_file_mlir = raii.enter_context(
            tempfile.NamedTemporaryFile(suffix=".mlir", mode="w")
        )
        print(
            module.operation.get_asm(enable_debug_info=True),
            file=temp_file_mlir,
        )
        temp_file_mlir.flush()

        temp_file_llvmir = raii.enter_context(
            tempfile.NamedTemporaryFile(suffix=".ll", mode="w")
        )
        subp.check_call(
            [
                "mlir-translate",
                "--mlir-to-llvmir",
                temp_file_mlir.name,
                "-o",
                temp_file_llvmir.name,
            ]
        )
        # subp.check_call(["cat", "out.ll"])
        subp.check_call(
            [
                "clang",
                "-o",
                out_path,
                temp_file_llvmir.name,
                "-Ldeps/spy/spy/libspy/build/native/release/",
                "-lspy",
            ]
        )


# def extra_egraph(expr):
#     import nbcc.egraph.grammar as sg
#     from nbcc.egraph.rules import VarAnn

#     match expr:
#         case sg.VarAnnotation(typename=typename, symbol=symbol, value=expr):
#             expr = yield expr
#             return VarAnn(typename, symbol, expr)
#         case _:
#             raise NotImplementedError


def middle_end(tu: TranslationUnit, fname: str):
    fqn, func_region = tu.get_function(fname)
    print(fqn, func_region)

    # memo = egraph_conversion(func_region, extra_handle=extra_egraph)
    memo = egraph_conversion(func_region)

    root = GraphRoot(memo[func_region])
    egraph = EGraph()
    egraph.let("root", root)
    # egraph.display()

    expand_struct_type(tu, egraph)

    egraph_optimize(egraph)

    cost, extracted = egraph_extraction(
        egraph, func_region, converter_class=ExtendEGraphToRVSDG
    )
    print("egraph extracted")
    print("cost", cost)

    print(extracted)

    [func] = [child for child in extracted._args if isinstance(child, rg.Func)]
    print(format_rvsdg(func))
    return func


def expand_struct_type(tu: TranslationUnit, egraph):
    from egglog import Ruleset

    from nbcc.egraph.rules import (
        create_ruleset_struct__get_field__,
        create_ruleset_struct__make__,
    )

    schedule = Ruleset("empty")
    for fqn_struct, w_obj_struct in tu._structs.items():
        print(fqn_struct, w_obj_struct)

        for fqn, w_obj in tu._builtins.items():
            if fqn_struct.fullname.startswith(fqn_struct.fullname):
                print("BUITIN", fqn)
                subname = fqn.parts[-1].name
                if subname == "__make__":
                    print("Add __make__")
                    schedule |= create_ruleset_struct__make__(w_obj)

                elif subname.startswith("__get_"):
                    print("Add field getter")
                    for i, k in enumerate(w_obj_struct.fields_w.keys()):
                        if subname == f"__get_{k}__":
                            schedule |= create_ruleset_struct__get_field__(
                                w_obj, i
                            )

    egraph.run(schedule.saturate())


if __name__ == "__main__":
    compile(sys.argv[1], sys.argv[2])
