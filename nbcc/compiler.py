import logging
import subprocess as subp
import sys
import tempfile
from contextlib import ExitStack

import sealir.rvsdg.grammar as rg
from egglog import EGraph, Vec
from mlir import ir
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import egraph_extraction
from sealir.rvsdg import format_rvsdg

from nbcc.egraph.conversion import ExtendEGraphToRVSDG
from nbcc.egraph.rules import egraph_optimize, egraph_convert_metadata
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


def middle_end(tu: TranslationUnit, fname: str):

    fi = tu.get_function(fname)
    print(fi.fqn, fi.region)

    memo = egraph_conversion(fi.region)

    root = GraphRoot(memo[fi.region])

    egraph = EGraph()
    egraph.let("root", root)
    egraph.let("mds", egraph_convert_metadata(fi.metadata, memo))

    expand_struct_type(tu, egraph)

    egraph_optimize(egraph)
    # egraph.display()

    cost, extracted = egraph_extraction(
        egraph, fi.region, converter_class=ExtendEGraphToRVSDG
    )
    print("egraph extracted")
    print("cost", cost)

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
