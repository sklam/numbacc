
import sys
from egglog import EGraph
from sealir.rvsdg import format_rvsdg
import sealir.rvsdg.grammar as rg
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import egraph_extraction

from nbcc.frontend import frontend, TranslationUnit
from nbcc.mlir_backend.backend import Backend
from nbcc.egraph.rules import egraph_optimize
from nbcc.egraph.conversion import ExtendEGraphToRVSDG

import logging
logging.disable(logging.INFO)

def compile(path: str):
    tu = frontend(path)

    rvsdg_ir = middle_end(tu, "main")
    be = Backend()
    module = be.lower(rvsdg_ir, ())
    print(module)
    module.operation.verify()

    module = be.run_passes(module)
    print("After optimization")
    print(module)

    make_binary(module)


def make_binary(module):
    from mlir import ir
    import subprocess as subp
    module: ir.Module
    with open("out.mlir", "w") as fout:
        print(module.operation.get_asm(enable_debug_info=True), file=fout)

    subp.check_call(["mlir-translate", "--mlir-to-llvmir", "out.mlir", "-o", "out.ll"])
    subp.check_call(["cat", "out.ll"])
    subp.check_call(["clang", "-o" "a.out", "out.ll", "-Ldeps/spy/spy/libspy/build/native/release/", "-lspy"])


def middle_end(tu: TranslationUnit, fname: str):
    fqn, func_region = tu.get_function(fname)
    print(fqn, func_region)
    memo = egraph_conversion(func_region)

    root = GraphRoot(memo[func_region])
    egraph = EGraph()
    egraph.let("root", root)
    # egraph.display()

    expand_struct_type(tu, egraph)


    egraph_optimize(egraph)

    cost, extracted = egraph_extraction(
        egraph, func_region,
        converter_class=ExtendEGraphToRVSDG
    )
    print('egraph extracted')
    print("cost", cost)

    print(extracted)

    [func] = [child for child in extracted._args if isinstance(child, rg.Func)]
    print(format_rvsdg(func))
    return func


def expand_struct_type(tu: TranslationUnit, egraph):
    from egglog import Ruleset
    from nbcc.egraph.rules import create_ruleset_struct__make__, create_ruleset_struct__get_field__
    schedule = Ruleset('empty')
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
                            schedule |= create_ruleset_struct__get_field__(w_obj, i)




    egraph.run(schedule.saturate())




if __name__ == "__main__":
    compile(sys.argv[1])
