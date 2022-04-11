import collections
import dataclasses
import functools
from typing import Dict

import sympy
from sympy import Expr
from sympy import Integer
from sympy import Symbol


@dataclasses.dataclass
class ZeroGuard:
    """
    An expression we should check equals zero.
    Guards are currently not checked.  Plan to add this later.
    """

    expr: sympy.Expr


class SizeVarAllocator(object):
    def __init__(self, prefix="s", zero_one_const=True):
        super().__init__()
        self.prefix = prefix
        self.val_to_var: Dict[int, Expr] = {0: Integer(0), 1: Integer(1)}
        self.var_to_val: Dict[Expr, int] = collections.OrderedDict()
        self.guards = []
        self.replacements = {}
        if not zero_one_const:
            self.val_to_var.clear()

    def guard_equals(self, left: sympy.Symbol, right: sympy.Symbol):
        expr = sympy.expand(left - right).subs(self.replacements)
        assert self.size_hint(expr) == 0
        free = list(expr.free_symbols)
        if len(free) == 0:
            assert expr == 0
            return
        elif len(free) in (1, 2, 3):
            # remove the largest of the guarded variables
            free.sort(key=self.size_hint)
            try:
                solutions = sympy.solve(expr, free[-1])
                if (
                    len(solutions) == 1
                    and solutions[0]
                    and "/" not in str(solutions[0])
                ):
                    self.replacements[free[-1]] = solutions[0]
            except NotImplementedError:
                pass

        self.guards.append(ZeroGuard(expr))

    def __getitem__(self, val):
        if val in self.val_to_var:
            return self.val_to_var[val]
        var = Symbol(f"{self.prefix}{len(self.var_to_val)}")
        self.val_to_var[val] = var
        self.var_to_val[var] = val
        return var

    def size_hint(self, expr: Expr):
        return int(expr.subs(self.var_to_val))

    def codegen(self, code, graph_inputs):
        """Assign all symbolic shapes to locals"""

        @functools.lru_cache(None)
        def sizeof(name):
            code.writeline(f"{name}_size = {name}.size()")
            return f"{name}_size"

        @functools.lru_cache(None)
        def strideof(name):
            code.writeline(f"{name}_stride = {name}.stride()")
            return f"{name}_stride"

        needed = set(map(str, self.var_to_val.keys())) - set(self.replacements.keys())

        for name, value in graph_inputs.items():
            shapes = value.get_size()
            for dim, shape in enumerate(shapes):
                shape = str(shape)
                if shape in needed:
                    needed.remove(shape)
                    code.writeline(f"{shape} = {sizeof(name)}[{dim}]")

        for name, value in graph_inputs.items():
            shapes = value.get_stride()
            for dim, shape in enumerate(shapes):
                shape = str(shape)
                if shape in needed:
                    needed.remove(shape)
                    code.writeline(f"{shape} = {strideof(name)}[{dim}]")

        assert not needed
