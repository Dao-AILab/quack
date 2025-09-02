# Copyright (c) 2025, Tri Dao.

import os
import pathlib
from functools import partial
from dataclasses import dataclass, fields

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import JitArgument
from cutlass.cutlass_dsl import NumericMeta


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [
            f for f in all_fields if not isinstance(f, (cutlass.Constexpr, NumericMeta))
        ]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {
            n: f for n, f in all_fields.items() if isinstance(f, (cutlass.Constexpr, NumericMeta))
        }
        non_constexpr_fields = {
            n: f
            for n, f in all_fields.items()
            if not isinstance(f, (cutlass.Constexpr, NumericMeta))
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class ArgumentsBase(JitArgument):
    def __c_pointers__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [
            f for f in all_fields if not isinstance(f, (cutlass.Constexpr, NumericMeta))
        ]
        c_ptrs = []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__c_pointers__"):
                c_ptrs.extend(obj.__c_pointers__())
        return c_ptrs

    def __get_mlir_types__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [
            f for f in all_fields if not isinstance(f, (cutlass.Constexpr, NumericMeta))
        ]
        types = []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__get_mlir_types__"):
                types.extend(obj.__get_mlir_types__())
        return types

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {
            n: f for n, f in all_fields.items() if isinstance(f, (cutlass.Constexpr, NumericMeta))
        }
        non_constexpr_fields = {
            n: f
            for n, f in all_fields.items()
            if not isinstance(f, (cutlass.Constexpr, NumericMeta))
        }
        # for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
        for name, field in non_constexpr_fields.items():
            # non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            # values = values[n_items:]
            n_items = 1
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


def load_cubin_module_data_patched(cubin_data, filepath):
    path = pathlib.Path(filepath)
    path.write_bytes(cubin_data)
    return load_cubin_module_data_og(cubin_data)


def cute_compile_patched(*args, **kwargs):
    """A patched version of cute.compile that dump the SASS to a file if CUTE_CUBIN_PATH is set."""
    if os.getenv("CUTE_CUBIN_PATH") is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = partial(
            load_cubin_module_data_patched, filepath=os.getenv("CUTE_CUBIN_PATH")
        )
    output = cute_compile_og(*args, **kwargs)
    if os.getenv("CUTE_CUBIN_PATH") is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = load_cubin_module_data_og
        if extract is not None:
            cubin_path = pathlib.Path(os.getenv("CUTE_CUBIN_PATH"))
            sass = extract(cubin_path, None)
            cubin_path.with_suffix(".annotated.sass").write_text(sass)
    return output
