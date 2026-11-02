"""Independent check that K = Q[t, y] / (F(t), y^2 - P1(t) y - P0(t)) is a field.

Method: K is a 16-dimensional Q-vector space with basis t^i, t^i y (i < 8).  For an element
theta of K, the characteristic polynomial of multiplication by theta is computed exactly
(fmpq_mat.charpoly) and factored over Q.  If F is irreducible and some theta has an irreducible
characteristic polynomial of degree 16, then Q(t, y) has degree 16 over Q, so the quadratic is
irreducible over Q(t) and K is a field.  (If the quadratic split, K would be Q(t) x Q(t) and
every characteristic polynomial would be a product of two degree-8 polynomials.)

Also reports the minimal polynomial of y itself (its characteristic polynomial factored), and
compares the characteristic polynomial of y + t with the file's univariate "f" (made monic).

Usage: indep_exact_field.py [path-to-json]
"""

import json
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indep_exact_core as core  # noqa: E402
from flint import fmpq, fmpq_poly, fmpq_mat  # noqa: E402


def P(*a):
    print(*a, flush=True)


def mult_matrix(R, theta):
    """Matrix of multiplication by theta on the basis t^i, t^i y of K (rows = images)."""
    n = R.F.degree()
    N = 2 * n
    rows = []
    basis = []
    for i in range(n):
        basis.append(R.elem([0] * i + [1]))
    for i in range(n):
        basis.append(R.elem([], [0] * i + [1]))
    for e in basis:
        img = e * theta
        row = [fmpq(0)] * N
        for k, c in enumerate(img.u.coeffs()):
            row[k] = c
        for k, c in enumerate(img.v.coeffs()):
            row[n + k] = c
        rows.append(row)
    return fmpq_mat(N, N, [rows[i][j] for i in range(N) for j in range(N)])


def monic(p):
    lc = p.coeffs()[-1]
    return p * (fmpq(1) / lc)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/sigma_rur_inner1_tower.json"
    t_start = time.time()
    P("indep_exact_field: start", time.strftime("%Y-%m-%d %H:%M:%S %Z"), "file:", path)
    d = json.load(open(path))
    if d.get("complete") is not True:
        P("file not complete; stopping")
        sys.exit(3)
    tower = d["tower"]
    F = core.poly_from_strings(tower["F"])
    P0 = core.poly_from_strings(tower["P0"])
    P1 = core.poly_from_strings(tower["P1"])
    R = core.KRing(F, P0, P1)
    n = F.degree()
    fac = F.factor()
    F_irred = len(fac[1]) == 1 and fac[1][0][1] == 1 and fac[1][0][0].degree() == n
    P("F degree", n, "irreducible over Q:", F_irred)

    y = R.elem([], [1])
    t = R.elem([0, 1])
    # minimal polynomial of y
    cy = mult_matrix(R, y).charpoly()
    cy_fac = cy.factor()
    P("char poly of y: factor degrees", [(q.degree(), e) for q, e in cy_fac[1]])
    for q, e in cy_fac[1]:
        P("  factor (monic):", monic(q), "multiplicity", e)
    # theta = y + k t for a few k
    field = False
    f_file = None
    if "f" in d:
        f_file = monic(core.poly_from_strings(d["f"]))
    for k in (1, 2, 3, -1):
        theta = y + R.from_rational(k) * t
        tk = time.time()
        c = mult_matrix(R, theta).charpoly()
        cf = c.factor()
        degs = [(q.degree(), e) for q, e in cf[1]]
        irred = len(cf[1]) == 1 and cf[1][0][1] == 1 and cf[1][0][0].degree() == 2 * n
        P(f"theta = y + ({k}) t: char poly degree {c.degree()}, factor degrees {degs}, irreducible: {irred}, {time.time() - tk:.1f} s")
        if k == 1 and f_file is not None:
            same = (monic(c) - f_file).is_zero()
            P("  char poly of y + t equals the file's f (both monic):", same)
            if not same:
                P("  difference (first coefficients):", str(monic(c) - f_file)[:200])
        if irred:
            field = True
    if F_irred and field:
        P("CONCLUSION: K is a field of degree", 2 * n, "over Q; y^2 - P1 y - P0 is irreducible over Q(t)")
    else:
        P("CONCLUSION: not established by this test (no theta tried had an irreducible char poly)")
    # degree of each free coordinate over Q (cheap, informative): count distinct min-poly degrees
    system = d["system"]
    uv = tower["uv"]
    degs = {}
    tk = time.time()
    for name in system["free"]:
        x = R.from_strings(uv[name][0], uv[name][1])
        c = mult_matrix(R, x).charpoly()
        cf = c.factor()
        # the minimal polynomial is the product of the distinct irreducible factors (K reduced)
        mdeg = sum(q.degree() for q, e in cf[1])
        degs[name] = mdeg
    hist = {}
    for v in degs.values():
        hist[v] = hist.get(v, 0) + 1
    P(f"degree over Q of the minimal polynomial of each free coordinate (histogram degree: count): {dict(sorted(hist.items()))}, {time.time() - tk:.1f} s")
    P("coordinates of degree 16:", [k for k, v in degs.items() if v == 16][:20], "...")
    P("coordinates of degree <= 8:", {k: v for k, v in degs.items() if v <= 8})
    P(f"indep_exact_field: done in {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
