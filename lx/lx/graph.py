"""Linear forms over residual dims (from Qwenputer's chipformer.dsl.graph, MIT, (c) 2026
Divyansh Agrawal)."""

from fractions import Fraction

Num = int | Fraction
ONE = 0  # dim id of the constant-one embedding feature


class Expr:
    """A linear form ``sum(coef * dim)``; the constant term lives on dim ONE."""

    __slots__ = ("terms",)

    def __init__(self, terms: dict[int, Num] | None = None):
        self.terms = {k: v for k, v in (terms or {}).items() if v != 0}

    def __add__(self, other: "Expr | Num") -> "Expr":
        other = as_expr(other)
        t = dict(self.terms)
        for k, v in other.terms.items():
            t[k] = t.get(k, 0) + v
        return Expr(t)

    __radd__ = __add__

    def __neg__(self) -> "Expr":
        return Expr({k: -v for k, v in self.terms.items()})

    def __sub__(self, other: "Expr | Num") -> "Expr":
        return self + (-as_expr(other))

    def __rsub__(self, other: "Expr | Num") -> "Expr":
        return as_expr(other) - self

    def __mul__(self, c: Num) -> "Expr":
        if not isinstance(c, (int, Fraction)):
            raise TypeError("Expr can only be scaled by a constant (use ffn/gate for products)")
        return Expr({k: v * c for k, v in self.terms.items()})

    __rmul__ = __mul__

    def __truediv__(self, c: int) -> "Expr":
        return self * Fraction(1, c)

    def __repr__(self) -> str:
        return " + ".join(f"{v}*d{k}" for k, v in self.terms.items()) or "0"


def as_expr(x: "Expr | Num") -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, Fraction)):
        return Expr({ONE: x})
    raise TypeError(f"cannot convert {type(x).__name__} to Expr")
