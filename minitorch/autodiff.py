from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol

import collections

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    if not 0 <= arg <= len(vals):
        raise ValueError(f"Argument {arg} is out of range")

    at = f(*vals)

    mod_vals = list(vals)
    mod_vals[arg] += epsilon
    near = f(*mod_vals)

    return (near - at) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...  # noqa

    @property
    def unique_id(self) -> int: ...  # noqa

    def is_leaf(self) -> bool: ...  # noqa

    def is_constant(self) -> bool: ...  # noqa

    @property
    def parents(self) -> Iterable["Variable"]: ...  # noqa

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...  # noqa


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    sort, seen = [], set()

    def walk(var: Variable) -> None:
        if var.unique_id in seen:
            return

        if not var.is_leaf():
            for input in var.parents:
                if not input.is_constant():
                    walk(input)

        seen.add(var.unique_id)

        if not var.is_constant():
            sort.append(var)

    walk(variable)
    return reversed(sort)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    diffs = collections.defaultdict(float, {variable.unique_id: deriv})

    for var in topological_sort(variable):
        if var.is_leaf():
            var.accumulate_derivative(diffs[var.unique_id])
        else:
            for v, d in var.chain_rule(diffs[var.unique_id]):
                diffs[v.unique_id] += d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the stored values"""
        return self.saved_values
