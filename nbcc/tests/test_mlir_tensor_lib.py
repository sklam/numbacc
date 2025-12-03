import os
import os.path
import tempfile
from contextlib import contextmanager
from ctypes import CDLL, byref, c_double
from pathlib import Path
from typing import Generator

import numpy as np
from mlir.runtime import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)

import nbcc
from nbcc.compiler import compile_shared_lib


example_dir = Path(os.path.dirname(nbcc.__file__)) / ".." / "examples"


@contextmanager
def make_temp_directory() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(delete=False) as dirpath:
        yield Path(dirpath)


@contextmanager
def compile_lib(filename: str, libname: str) -> str:
    path = example_dir / filename
    assert path.exists()
    with make_temp_directory() as dir:
        outpath = dir / libname
        compile_shared_lib(str(path), str(outpath))
        yield outpath


def test_has_examples():
    assert example_dir.exists()


NELEM = 400
benchmark_config = dict(rounds=100, iterations=3)


def test_mlir_tensor_lib_add():
    with compile_lib("mlir_tensor_lib.spy", "lib_mlir_tensor.so") as libname:
        lib = CDLL(libname)
        func = getattr(
            lib, "_mlir_ciface_spy_mlir_tensor_lib$export_tensor_f64_add"
        )
        print(func)

        # RUN
        memref_1d_f64 = make_nd_memref_descriptor(1, c_double)

        A = np.arange(NELEM, dtype=np.float64) / NELEM
        B = np.arange(NELEM, dtype=np.float64) / NELEM

        argA = get_ranked_memref_descriptor(A)
        argB = get_ranked_memref_descriptor(B)

        out_memref = (memref_1d_f64 * 1)()
        args = [out_memref, byref(argA), byref(argB)]
        func(*args)

        output = ranked_memref_to_numpy(out_memref)
        print(output)

        np.testing.assert_allclose(output, A + B)


def test_mlir_tensor_lib_arrayexpr():
    with compile_lib("mlir_tensor_lib.spy", "lib_mlir_tensor.so") as libname:
        lib = CDLL(libname)
        func = getattr(
            lib, "_mlir_ciface_spy_mlir_tensor_lib$export_tensor_f64_arrayexpr"
        )
        print(func)

        # RUN
        memref_1d_f64 = make_nd_memref_descriptor(1, c_double)

        A = np.arange(NELEM, dtype=np.float64) / NELEM
        B = np.arange(NELEM, dtype=np.float64) / NELEM
        C = np.arange(NELEM, dtype=np.float64) / NELEM

        argA = get_ranked_memref_descriptor(A)
        argB = get_ranked_memref_descriptor(B)
        argC = get_ranked_memref_descriptor(C)

        out_memref = (memref_1d_f64 * 1)()
        args = [out_memref, byref(argA), byref(argB), byref(argC)]
        func(*args)

        output = ranked_memref_to_numpy(out_memref)
        print(output)

        np.testing.assert_allclose(output, (A + B) * (C + A))


def test_bench_baseline_add(benchmark):

    A = np.arange(NELEM, dtype=np.float64) / NELEM
    B = np.arange(NELEM, dtype=np.float64) / NELEM

    fn = lambda a, b: a + b
    fn(A, B)  # warm up

    benchmark.pedantic(fn, args=(A, B), **benchmark_config)


def test_bench_mlir_tensor_lib_add(benchmark):
    with compile_lib("mlir_tensor_lib.spy", "lib_mlir_tensor.so") as libname:
        lib = CDLL(libname)
        func = getattr(
            lib, "_mlir_ciface_spy_mlir_tensor_lib$export_tensor_f64_add"
        )
        print(func)

        # RUN

        memref_1d_f64 = make_nd_memref_descriptor(1, c_double)

        A = np.arange(NELEM, dtype=np.float64) / NELEM
        B = np.arange(NELEM, dtype=np.float64) / NELEM

        argA = get_ranked_memref_descriptor(A)
        argB = get_ranked_memref_descriptor(B)

        out_memref = (memref_1d_f64 * 1)()
        args = [out_memref, byref(argA), byref(argB)]
        func(*args)

        output = ranked_memref_to_numpy(out_memref)
        print(output)

        np.testing.assert_allclose(output, A + B)

        def cleanup(*args):
            ranked_memref_to_numpy(args[0])  # cleanup

        # Requires pytest-benchmark >= 5.2.0 for teardown
        benchmark.pedantic(
            func, args=args, teardown=cleanup, **benchmark_config
        )


def test_bench_baseline_arrayexpr(benchmark):

    A = np.arange(NELEM, dtype=np.float64) / NELEM
    B = np.arange(NELEM, dtype=np.float64) / NELEM
    C = np.arange(NELEM, dtype=np.float64) / NELEM

    fn = lambda a, b, c: (a + b) * (c + a)
    fn(A, B, C)  # warm up

    benchmark.pedantic(fn, args=(A, B, C), **benchmark_config)


def test_bench_mlir_tensor_lib_arrayexpr(benchmark):
    with compile_lib("mlir_tensor_lib.spy", "lib_mlir_tensor.so") as libname:
        lib = CDLL(libname)
        func = getattr(
            lib, "_mlir_ciface_spy_mlir_tensor_lib$export_tensor_f64_arrayexpr"
        )
        print(func)

        # RUN

        memref_1d_f64 = make_nd_memref_descriptor(1, c_double)

        A = np.arange(NELEM, dtype=np.float64) / NELEM
        B = np.arange(NELEM, dtype=np.float64) / NELEM
        C = np.arange(NELEM, dtype=np.float64) / NELEM

        argA = get_ranked_memref_descriptor(A)
        argB = get_ranked_memref_descriptor(B)
        argC = get_ranked_memref_descriptor(C)

        out_memref = (memref_1d_f64 * 1)()
        args = [out_memref, byref(argA), byref(argB), byref(argC)]
        func(*args)

        output = ranked_memref_to_numpy(out_memref)
        print(output)

        np.testing.assert_allclose(output, (A + B) * (C + A))

        def cleanup(*args):
            ranked_memref_to_numpy(args[0])  # cleanup

        # Requires pytest-benchmark >= 5.2.0 for teardown
        benchmark.pedantic(
            func, args=args, teardown=cleanup, **benchmark_config
        )
