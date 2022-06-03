import builtins
import collections
import copy
import functools
import inspect
import itertools
import math
import operator
import types
import warnings
from typing import Dict
from typing import Optional
from typing import Set

import numpy
import torch

from . import config


def make_function_id_set(lazy_initializer):
    """
    Track a set of `id()`s of objects which are either allowed or not
    allowed to go into the generated FX graph.  Use to test for torch.*,
    numpy.*, builtins.*, etc.

    Support user modification to permit customization of what can be
    added to the graph and what will cause a graph break.
    """

    class FunctionIdSet:
        function_ids: Optional[Set[int]] = None
        function_names: Optional[Dict[int, str]] = None

        def __call__(self):
            if self.function_ids is None:
                value = lazy_initializer()
                if isinstance(value, dict):
                    self.function_ids = set(value.keys())
                    self.function_names = value
                else:
                    assert isinstance(value, set)
                    self.function_ids = value
            return self.function_ids

        def get_name(self, idx: int, default: str):
            self()  # lazy init
            return self.function_names.get(idx, default)

        def add(self, idx: int):
            self()  # lazy init
            self.function_ids.add(idx)

        def remove(self, idx: int):
            if idx in self():
                self.function_ids.remove(idx)

        def __contains__(self, idx: int):
            return idx in self()

    return FunctionIdSet()


@make_function_id_set
def _disallowed_function_ids():
    remove = [
        True,
        False,
        None,
        collections.OrderedDict,
        copy.copy,
        copy.deepcopy,
        inspect.signature,
        torch.autocast_decrement_nesting,
        torch.autocast_increment_nesting,
        torch.autograd.grad,
        torch.clear_autocast_cache,
        torch.cuda.current_device,
        torch.distributions.constraints.is_dependent,
        torch.distributions.normal.Normal,
        torch.inference_mode,
        torch.set_anomaly_enabled,
        torch.set_autocast_cache_enabled,
        torch.set_autocast_cpu_dtype,
        torch.set_autocast_cpu_enabled,
        torch.set_autocast_enabled,
        torch.set_autocast_gpu_dtype,
        torch.autograd.profiler.profile,
        warnings.warn,
    ]
    return {id(x) for x in remove}


@make_function_id_set
def _allowed_function_ids():
    """
    Walk torch.* and get the ids of all the stuff in it
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    torch.distributions.Distribution.set_default_validate_args(False)
    torch_object_ids = dict()

    def _find_torch_objects(module):
        if any(
            module.__name__.startswith(mod_name)
            for mod_name in config.allowed_functions_module_string_ignorelist
        ):
            return
        torch_object_ids[id(module)] = module.__name__
        for name, obj in list(module.__dict__.items()):
            if id(obj) not in torch_object_ids:
                if isinstance(obj, types.ModuleType):
                    if obj.__name__.startswith("torch."):
                        torch_object_ids[id(obj)] = f"{module.__name__}.{name}"
                        _find_torch_objects(obj)
                elif inspect.getmodule(obj) and (
                    inspect.getmodule(obj).__name__.startswith("torch")
                    or inspect.getmodule(obj).__name__.startswith("math")
                ):
                    torch_object_ids[id(obj)] = f"{module.__name__}.{name}"

    _find_torch_objects(torch)
    _find_torch_objects(math)

    for idx in _disallowed_function_ids():
        if idx in torch_object_ids:
            del torch_object_ids[idx]

    return torch_object_ids


@make_function_id_set
def _builtin_function_ids():
    rv = {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and callable(v)
    }
    rv.update(
        {
            id(v): f"operator.{k}"
            for k, v in operator.__dict__.items()
            if not k.startswith("_") and callable(v)
        }
    )
    rv.update(
        {id(v): f"functools.{v.__name__}" for v in (itertools.chain, itertools.islice)}
    )
    rv[id(functools.reduce)] = "functools.reduce"
    return rv


@make_function_id_set
def _numpy_function_ids():
    rv = dict()
    for mod in (numpy, numpy.random):
        rv.update(
            {
                id(v): f"{mod.__name__}.{k}"
                for k, v in mod.__dict__.items()
                if callable(v)
                and (getattr(v, "__module__", None) or mod.__name__) == mod.__name__
            }
        )
    return rv


def is_allowed(obj):
    """Is this safe to trace like torch.add ?"""
    return id(obj) in _allowed_function_ids


def is_disallowed(obj):
    """Is this safe to trace like torch.add ?"""
    return id(obj) in _disallowed_function_ids


def torch_get_name(obj, default):
    """Convert a torch.* funcion to a string"""
    return _allowed_function_ids.get_name(id(obj), default)


def is_builtin(obj):
    return id(obj) in _builtin_function_ids


def is_numpy(obj):
    return isinstance(obj, numpy.ndarray) or id(obj) in _numpy_function_ids
