"""Independent exact verifier, univariate form ("f", "g") of results/sigma_rur_inner1_tower.json.

x_j = g_j(theta) / f'(theta) in Q[theta]/(f).  The inverse of f'(theta) is computed by the
extended Euclidean algorithm (fmpq_poly.xgcd); the tableau is built over Q[theta]/(f) with the
same generic order-condition code as the tower check (indep_exact_core), and every residual must
reduce to the zero polynomial.  Then the real roots of f are isolated by Sturm sequences and the
tableau is evaluated in arb at each real root and compared with the tower embedding chosen in
indep_exact_tower (same root-selection rules), to confirm that the two forms describe the same
real method.

Usage: indep_exact_univariate.py [path-to-json] [--skip-exact]
"""

import json
import math
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indep_exact_core as core  # noqa: E402
import indep_exact_tower as tw  # noqa: E402
import flint  # noqa: E402
from flint import fmpq, fmpq_poly, arb, arb_poly  # noqa: E402

P = tw.P


def main():
    args = sys.argv[1:]
    path = "results/sigma_rur_inner1_tower.json"
    skip_exact = False
    only_root = None
    bits = 7000
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--skip-exact":
            skip_exact = True
        elif a == "--only-root":
            only_root = int(args[i + 1])
            i += 1
        elif a == "--bits":
            bits = int(args[i + 1])
            i += 1
        else:
            path = a
        i += 1
    t_start = time.time()
    P("indep_exact_univariate: start", tw.stamp(), "file:", path)
    d = tw.load(path)
    system = d["system"]
    s = int(system["s"])
    p_order = int(system["p"])
    free = list(system["free"])
    if "f" not in d or "g" not in d:
        P("no univariate form (keys f, g) in the file; nothing to do")
        return
    fc = tw.fr_poly(d["f"])
    f = fmpq_poly([core.frac_to_fmpq(c) for c in fc])
    P(f"f degree {f.degree()}, g entries {len(d['g'])}, free {len(free)}")
    missing = [n for n in free if n not in d["g"]]
    if missing:
        P("free coordinates without a g entry:", missing)
        sys.exit(4)
    f_fac = f.factor()
    P("f factor degrees over Q:", [(q.degree(), e) for q, e in f_fac[1]],
      " irreducible:", len(f_fac[1]) == 1 and f_fac[1][0][1] == 1 and f_fac[1][0][0].degree() == f.degree())
    fp = f.derivative()
    gg, sgcd, tgcd = fp.xgcd(f)
    P("gcd(f', f) =", gg)
    if gg.degree() != 0:
        P("f' is not invertible modulo f (f not squarefree); cannot form g/f'")
        sys.exit(5)
    inv_fp = (sgcd * (fmpq(1) / gg.coeffs()[0])) % f
    assert ((inv_fp * fp) % f - fmpq_poly([1])).is_zero()
    R = core.KRing(f, fmpq_poly([]), fmpq_poly([]))

    def lookup(name):
        gj = core.poly_from_strings(d["g"][name])
        return R.elem((gj * inv_fp) % f)

    A, b, rep = core.build_tableau(system, lookup, R.from_rational)
    P("tableau built over Q[theta]/(f):", rep)

    if not skip_exact:
        P("\n== exact order conditions in Q[theta]/(f) ==")
        t0 = time.time()
        oc = core.OrderConditions(A, b, R.from_rational)

        def progress(k, nt, nz, secs):
            P(f"order {k}: {nt} trees, non-zero residuals: {nz}, {secs:.1f} s")

        res = oc.residuals_pruned(p_order, progress)
        all_zero = all(r.is_zero() for _, _, r in res)
        summ = core.summarize_residuals(res)
        for k in sorted(summ):
            if summ[k][1]:
                P(f"  FAIL order {k}: {summ[k][1]} of {summ[k][0]}: {summ[k][2][:20]}")
        P(f"exact univariate check: {len(res)} conditions, all zero: {all_zero}, {time.time() - t0:.1f} s")

    # ---- real roots of f and numerical agreement with the tower embedding ----
    P("\n== real roots of f and agreement with the tower embedding ==")
    t0 = time.time()
    roots, nreal = tw.isolate_real_roots(fc)
    P(f"f has {nreal} distinct real roots (Sturm)")
    BITS = bits
    flint.ctx.prec = bits + 1000
    P(f"bisection bits {BITS}, arb precision {flint.ctx.prec} bits")

    def eval_uni(e, th):
        if e.u.degree() < 0:
            return arb(0)
        return arb_poly([arb(c) for c in e.u.coeffs()])(th)

    # tower embedding (same selection as indep_exact_tower)
    tower = d["tower"]
    Fc = tw.fr_poly(tower["F"])
    P0c = tw.fr_poly(tower["P0"])
    P1c = tw.fr_poly(tower["P1"])
    # D = P1^2 + 4 P0 as Fraction coefficient list
    P1q = fmpq_poly([core.frac_to_fmpq(c) for c in P1c])
    P0q = fmpq_poly([core.frac_to_fmpq(c) for c in P0c])
    Dq = P1q * P1q + 4 * P0q
    Dc = [core.fmpq_to_frac(c) for c in Dq.coeffs()]
    RT = core.KRing(fmpq_poly([core.frac_to_fmpq(c) for c in Fc]), P0q, P1q)
    uv = tower["uv"]
    AT, bT, _ = core.build_tableau(system, lambda n: RT.from_strings(uv[n][0], uv[n][1]), RT.from_rational)
    troots, _ = tw.isolate_real_roots(Fc)
    lane_t0 = Fraction(tw.LANE_T0)
    lane_y0 = Fraction(tw.LANE_Y0)
    tower_vals = None
    for a, bnd in troots:
        lo, hi = tw.refine_root(Fc, a, bnd, BITS)
        if not (lo - Fraction(1, 10 ** 15) <= lane_t0 <= hi + Fraction(1, 10 ** 15)):
            continue
        tball = tw.interval_to_arb(lo, hi)
        Dball = tw.arb_poly_from_fr(Dc)(tball)
        if not tw.arb_lower_bound(Dball) > 0:
            continue
        P1ball = tw.arb_poly_from_fr(P1c)(tball)
        for sgn in (+1, -1):
            yball = (P1ball + sgn * Dball.sqrt()) / 2
            if tw.arb_lower_bound(yball) - Fraction(1, 10 ** 8) <= lane_y0 <= tw.arb_upper_bound(yball) + Fraction(1, 10 ** 8):
                def evK(e):
                    val = arb(0)
                    if e.u.degree() >= 0:
                        val += arb_poly([arb(c) for c in e.u.coeffs()])(tball)
                    if e.v.degree() >= 0:
                        val += arb_poly([arb(c) for c in e.v.coeffs()])(tball) * yball
                    return val
                tower_vals = ([[evK(AT[i][j]) for j in range(s)] for i in range(s)], [evK(bT[i]) for i in range(s)])
                P(f"tower embedding: t0 = {tball.str(25, radius=True)}, sign {sgn:+d}, y0 = {yball.str(25, radius=True)}")
    if tower_vals is None:
        P("could not reproduce the tower embedding; comparing nothing")
    best = None
    for idx, (a, bnd) in enumerate(roots):
        if only_root is not None and idx != only_root:
            continue
        lo, hi = tw.refine_root(fc, a, bnd, BITS)
        th = tw.interval_to_arb(lo, hi)
        A_u = [[eval_uni(A[i][j], th) for j in range(s)] for i in range(s)]
        b_u = [eval_uni(b[i], th) for i in range(s)]
        c_u = [sum(A_u[i][j] for j in range(s)) for i in range(s)]
        line = f"root {idx}: theta = {th.str(20, radius=True)}, c3 = {c_u[2].str(20, radius=True)}, c12 = {c_u[11].str(20, radius=True) if s > 11 else '-'}"
        if tower_vals is not None:
            diffs = []
            for i in range(s):
                for j in range(s):
                    diffs.append(abs(A_u[i][j] - tower_vals[0][i][j]))
                diffs.append(abs(b_u[i] - tower_vals[1][i]))
            dmax = max(tw.arb_upper_bound(x) for x in diffs)
            line += f", max |A,b difference to tower| <= {float(dmax):.3e}"
            if best is None or dmax < best[0]:
                best = (dmax, idx)
        P(line)
    if best is not None:
        P(f"closest univariate real root to the tower embedding: root {best[1]}, max difference bound {float(best[0]):.3e}")
    P(f"real-root comparison done in {time.time() - t0:.1f} s")
    P(f"indep_exact_univariate: done in {time.time() - t_start:.1f} s at {tw.stamp()}")


if __name__ == "__main__":
    main()
