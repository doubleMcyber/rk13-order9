"""Butcher order conditions for explicit Runge-Kutta methods, in exact or arbitrary-precision arithmetic.

For a rooted tree t with children t_1..t_m, the stage vector is u(t) = prod_k (A u(t_k)) (elementwise),
with u(leaf) = (1, ..., 1). The order condition is  b . u(t) = 1 / gamma(t),  where
gamma(t) = |t| * prod_k gamma(t_k). A method has order p iff the condition holds for every tree with
at most p nodes (Butcher). The number of trees with n nodes is OEIS A000081: 1, 1, 2, 4, 9, 20, 48, 115,
286, 719, so order 9 has 486 conditions and order 10 has 1,205.

Entries of A and b may be Fractions (exact), mpmath mpf (high precision) or floats.
"""
from fractions import Fraction
from functools import lru_cache


def _key(t):
    return (_size(t), tuple(_key(c) for c in t))


@lru_cache(maxsize=None)
def _size(t):
    return 1 + sum(_size(c) for c in t)


@lru_cache(maxsize=None)
def trees(n):
    """All rooted trees with n nodes, each a tuple of child trees in nonincreasing canonical order."""
    if n == 1:
        return ((),)
    return tuple(_forests(n - 1, None))


def _forests(m, bound):
    if m == 0:
        yield ()
        return
    for k in range(m, 0, -1):
        for t in trees(k):
            kt = _key(t)
            if bound is not None and kt > bound:
                continue
            for rest in _forests(m - k, kt):
                yield (t,) + rest


@lru_cache(maxsize=None)
def gamma(t):
    g = _size(t)
    for c in t:
        g *= gamma(c)
    return g


def residuals(A, b, p, one=1):
    """Return a list of (order, tree, b.u(t) - 1/gamma(t)) for every tree with at most p nodes.

    `one` sets the number type of the constant 1 (pass Fraction(1) or mpmath.mpf(1))."""
    s = len(b)
    memo = {}

    def u(t):
        if t in memo:
            return memo[t]
        v = [one] * s
        for c in t:
            uc = u(c)
            Auc = [sum(A[i][j] * uc[j] for j in range(i)) for i in range(s)]  # explicit: j < i
            v = [v[i] * Auc[i] for i in range(s)]
        memo[t] = v
        return v

    out = []
    for n in range(1, p + 1):
        for t in trees(n):
            phi = sum(b[i] * u(t)[i] for i in range(s))
            out.append((n, t, phi - one / gamma(t)))
    return out


def max_residual_by_order(A, b, p, one=1):
    worst = {}
    for n, _, r in residuals(A, b, p, one):
        worst[n] = max(worst.get(n, 0), abs(r))
    return worst


def check_row_sums(A, c):
    """Max |sum_j A[i][j] - c[i]|; the tree conditions above assume c_i = sum_j a_ij."""
    return max(abs(sum(A[i][j] for j in range(len(A))) - c[i]) for i in range(len(A)))


if __name__ == "__main__":
    counts = [len(trees(n)) for n in range(1, 11)]
    print("trees per order 1..10:", counts, "cumulative to 9:", sum(counts[:9]), "to 10:", sum(counts))
