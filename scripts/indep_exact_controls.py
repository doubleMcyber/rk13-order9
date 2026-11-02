"""Independent exact verifier, controls (no data file needed).

1. Tree counts by size against A000081 (1, 1, 2, 4, 9, 20, 48, 115, 286; 486 trees up to 9).
2. Classical RK4 in exact Fractions: all 8 conditions up to order 4 hold, at least one of the
   9 order-5 conditions fails.
3. Self-test of the K arithmetic: y*y == P1*y + P0, (t^k) reduces modulo F, and the RK4 check
   repeated with every coefficient embedded in a non-trivial K gives the same verdicts.
"""

import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indep_exact_core as core  # noqa: E402


def main():
    t0 = time.time()
    print("indep_exact_controls: start", time.strftime("%Y-%m-%d %H:%M:%S %Z"), flush=True)

    counts, total = core.check_tree_counts(9)
    print("tree counts by size 1..9:", counts, "total:", total, flush=True)
    assert total == 486
    # a few named trees as a spot check of the canonical form
    trees3 = core.trees_of_size(3)
    print("size-3 trees:", [core.tree_str(t) for t in trees3],
          "densities:", [core.density(t) for t in trees3], flush=True)
    trees4 = core.trees_of_size(4)
    print("size-4 trees:", [core.tree_str(t) for t in trees4],
          "densities:", [core.density(t) for t in trees4], flush=True)
    # densities of all 486 trees: sum of 1/gamma over trees of each order should be 1/n!?  No:
    # the known identity is sum over trees of order n of alpha(t)/gamma(t) = 1/n!, which needs
    # the symmetry factor; instead check the maximal density n! (the tall tree) is present.
    for n in range(1, 10):
        ds = [core.density(t) for t in core.trees_of_size(n)]
        import math
        assert max(ds) == math.factorial(n), (n, max(ds))
        assert min(ds) == n, (n, min(ds))
    print("density range per size ok: min = n (bushy tree), max = n! (tall tree)", flush=True)

    # ---- RK4 over Fractions ----
    h = Fraction(1, 2)
    A = [[0, 0, 0, 0], [h, 0, 0, 0], [0, h, 0, 0], [0, 0, 1, 0]]
    A = [[Fraction(x) for x in row] for row in A]
    b = [Fraction(1, 6), Fraction(1, 3), Fraction(1, 3), Fraction(1, 6)]
    oc = core.OrderConditions(A, b, Fraction)
    res = oc.residuals_up_to(5)
    summ = core.summarize_residuals(res)
    for k in sorted(summ):
        n, nf, fl = summ[k]
        print(f"RK4 order {k}: {n} trees, {nf} failing", ("" if nf == 0 else f"failing: {fl}"),
              flush=True)
    assert all(summ[k][1] == 0 for k in (1, 2, 3, 4)), "RK4 must satisfy orders 1..4"
    assert summ[5][1] >= 1, "RK4 must fail at least one order-5 condition"
    fails5 = [(core.tree_str(t), r) for t, k, r in res if k == 5 and r != 0]
    print("RK4 order-5 residuals (exact):", fails5, flush=True)
    print("RK4 control: PASS (orders 1..4 exact, order 5 fails)", flush=True)

    # ---- K arithmetic self-test ----
    # F = t^3 - 2 (irreducible), P1 = t, P0 = t^2 + 1: y^2 = t y + t^2 + 1
    R = core.KRing([-2, 0, 0, 1], [1, 0, 1], [0, 1])
    y = R.elem([], [1])
    t = R.elem([0, 1])
    lhs = y * y
    rhs = R.elem([0, 1]) * y + R.elem([1, 0, 1])
    assert (lhs - rhs).is_zero(), "y*y != P1 y + P0"
    t3 = t * t * t
    assert (t3 - R.from_rational(2)).is_zero(), "t^3 != 2 in Q[t]/(t^3-2)"
    # associativity and commutativity spot check with mixed elements
    a = R.elem([1, 2, 3], [4, 5, 6])
    bb = R.elem([Fraction(1, 7), -1, 0], [2, 0, -3])
    c = R.elem([0, 0, 1], [1, 1, 1])
    assert ((a * bb) * c - a * (bb * c)).is_zero()
    assert (a * bb - bb * a).is_zero()
    assert (a * (bb + c) - (a * bb + a * c)).is_zero()
    print("K arithmetic self-test: PASS", flush=True)

    # RK4 embedded in K (all coefficients rational, so the y-part must stay zero)
    AK = [[R.from_rational(x) for x in row] for row in A]
    bK = [R.from_rational(x) for x in b]
    ocK = core.OrderConditions(AK, bK, R.from_rational)
    resK = ocK.residuals_up_to(5)
    summK = core.summarize_residuals(resK)
    assert all(summK[k][1] == 0 for k in (1, 2, 3, 4)) and summK[5][1] == summ[5][1]
    for (t1, k1, r1), (t2, k2, r2) in zip(res, resK):
        assert t1 == t2
        assert r2.v.is_zero()
        assert (r2 - R.from_rational(r1)).is_zero()
    print("RK4 over K agrees with RK4 over Fractions tree by tree: PASS", flush=True)

    print(f"indep_exact_controls: done in {time.time() - t0:.2f} s", flush=True)


if __name__ == "__main__":
    main()
