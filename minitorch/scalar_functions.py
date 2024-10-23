from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Invoke the function with the passed arguments storing the passed `vals`"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Invoke negation function saving arguments into context as necessary"""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute negation derivative on arguments in context, scaled by arbitrary input"""
        return -d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Invoke inverse function saving arguments into context as necessary"""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute inverse derivative on arguments in context, scaled by arbitrary input"""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Invoke logarithm function saving arguments into context as necessary"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute logarithmic derivative on arguments in context, scaled by arbitrary input"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = e ^ x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Invoke exponentiation function saving arguments into context as necessary"""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute exponential derivative on arguments in context, scaled by arbitrary input"""
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Invoke ReLU function saving arguments into context as necessary"""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute ReLU derivative on arguments in context, scaled by arbitrary input"""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Invoke sigmoid function saving arguments into context as necessary"""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute sigmoid derivative on arguments in context, scaled by arbitrary input"""
        (a,) = ctx.saved_values
        return (
            (operators.inv(1 + operators.exp(-a)) ** 2) * operators.exp(-a) * d_output
        )


class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Invoke addition function saving arguments into context as necessary"""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute addition derivative on arguments in context, scaled by arbitrary input"""
        return d_output, d_output


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Invoke multiplication function saving arguments into context as necessary"""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute multiplication derivative on arguments in context, scaled by arbitrary input"""
        (a, b) = ctx.saved_values
        return b * d_output, a * d_output


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Invoke equal to function saving arguments into context as necessary"""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute equal to derivative on arguments in context, scaled by arbitrary input"""
        return 0.0


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Invoke less than function saving arguments into context as necessary"""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute less than derivative on arguments in context, scaled by arbitrary input"""
        return 0.0
