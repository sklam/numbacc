from dataclasses import dataclass
from typing import Callable


def module_pipeline(*passes) -> str:
    parts = []
    for ps in passes:
        match ps:
            case ModulePass() as mp:
                pss = mp.get_unwrapped()
                options = mp.get_options()
            case FunctionPass() as fp:
                pss = fp.get_wrapped()
                options = fp.get_options()
            case _:
                raise ValueError(ps)
        if not options:
            parts.append(pss)
        else:
            parts.append(f"{pss}{{{options}}}")
    return f"{ModulePass.anchor}({','.join(parts)})"


@dataclass
class PassOption:
    name: str
    value_ctor: Callable[[object], str]


bool_ctor = lambda x: str(int(bool(x)))


class Pass:
    anchor: str
    passname: str
    options: list[str]
    _pass_options: dict[str, PassOption]

    def __init_subclass__(cls):
        cls._pass_options = {
            k: v for k, v in cls.__dict__.items() if isinstance(v, PassOption)
        }

    def __init__(self, **kwargs):
        self.options = options = []
        for k, v in kwargs.items():
            field = self._pass_options[k]
            options.append(f"{field.name}={field.value_ctor(v)}")

    def get_unwrapped(self) -> str:
        return self.passname

    def get_wrapped(self) -> str:
        return f"{self.anchor}({self.passname})"

    def get_options(self) -> str:
        return " ".join(self.options)


class ModulePass(Pass):
    anchor: str = "builtin.module"


class FunctionPass(Pass):
    anchor: str = "func.func"


class Canonicalize(ModulePass):
    passname = "canonicalize"


class TransformInterpreter(ModulePass):
    passname = "transform-interpreter"


class LowerVectorMask(FunctionPass):
    passname = "lower-vector-mask"


class OneShotBufferize(ModulePass):
    passname = "one-shot-bufferize"
    bufferize_function_boundaries = PassOption(
        "bufferize-function-boundaries", bool_ctor
    )


class OwnershipBasedBufferDeallocation(ModulePass):
    passname = "ownership-based-buffer-deallocation"


class BufferizationLowerDeallocations(ModulePass):
    passname = "bufferization-lower-deallocations"


class ConvertBufferizationToMemRef(ModulePass):
    passname = "convert-bufferization-to-memref"


class BufferDeallocationSimplification(ModulePass):
    passname = "buffer-deallocation-simplification"


class BufferHoisting(FunctionPass):
    passname = "buffer-hoisting"


class BufferLoopHoisting(FunctionPass):
    passname = "buffer-loop-hoisting"


class FoldMemRefAliasOps(ModulePass):
    passname = "fold-memref-alias-ops"


class FoldTensorSubsetOps(ModulePass):
    passname = "fold-tensor-subset-ops"


class PromoteBuffersToStack(FunctionPass):
    passname = "promote-buffers-to-stack"


class Mem2Reg(ModulePass):
    passname = "mem2reg"


class LoopInvariantCodeMotion(ModulePass):
    passname = "loop-invariant-code-motion"


class ConvertVectorToSCF(ModulePass):
    passname = "convert-vector-to-scf"


class ConvertVectorToLLVM(ModulePass):
    passname = "convert-vector-to-llvm"
    enable_arm_neon = PassOption("enable-arm-neon", bool_ctor)
    enable_arm_sve = PassOption("enable-arm-sve", bool_ctor)


class FinalizeMemRefToLLVM(ModulePass):
    passname = "finalize-memref-to-llvm"


class ConvertArithToLLVM(ModulePass):
    passname = "convert-arith-to-llvm"


class ConvertSCFToCF(ModulePass):
    passname = "convert-scf-to-cf"


class ConvertCFToLLVM(ModulePass):
    passname = "convert-cf-to-llvm"


class ConvertUbToLLVM(ModulePass):
    passname = "convert-ub-to-llvm"


class ConvertFuncToLLVM(ModulePass):
    passname = "convert-func-to-llvm"


class ConvertMathToLibM(ModulePass):
    passname = "convert-math-to-libm"


class ConvertIndexToLLVM(ModulePass):
    passname = "convert-index-to-llvm"


class ReconileUnrealizedCasts(ModulePass):
    passname = "reconcile-unrealized-casts"


class ConvertLinalgToAffineLoops(ModulePass):
    passname = "convert-linalg-to-affine-loops"


class ConvertLinalgToLoops(ModulePass):
    passname = "convert-linalg-to-loops"


class LowerAffine(FunctionPass):
    passname = "lower-affine"


class AffineSimplifyStructures(FunctionPass):
    passname = "affine-simplify-structures"


class MemRefExpand(FunctionPass):
    passname = "memref-expand"


class NormalizeMemRefs(ModulePass):
    passname = "normalize-memrefs"


class ExpandStridedMetadata(ModulePass):
    passname = "expand-strided-metadata"
