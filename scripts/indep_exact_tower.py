"""Independent exact verifier for the tower form of results/sigma_rur_inner1_tower.json.

Steps (all in exact arithmetic unless stated):
  1. load the file, require "complete": true, build the 13-stage tableau over
     K = Q[t, y] / (F(t), y^2 - P1(t) y - P0(t)) from "system" and "tower" (not from "f", "g");
  2. irreducibility: factor F over Q, gcd(F, F'); characteristic polynomial of multiplication
     by y on K (16-dimensional over Q) factored over Q;
  3. all 486 order conditions up to order 9 as residuals in K (zero means both parts zero);
  4. control: perturb one free coordinate by 1e-6 in the constant term of u and recheck;
  5. real embedding: Sturm sequence in Fractions isolates every real root of F, exact bisection
     narrows the chosen root, the discriminant P1^2 + 4 P0 is shown positive on the isolating
     interval by exact rational bounds and by arb, y0 is chosen by sign, the tableau is
     evaluated in arb (rigorous balls) and printed to 30 digits;
  6. sanity: all 486 residuals in mpmath at 60 digits at that embedding.

Usage: indep_exact_tower.py [path-to-json] [--perturb NAME] [--skip-exact]
"""

import json
import math
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indep_exact_core as core  # noqa: E402
import flint  # noqa: E402
from flint import fmpq, fmpq_poly, fmpz_poly, fmpq_mat, arb, arb_poly  # noqa: E402

LANE_T0 = "0.8769593826547598162"
LANE_Y0 = "0.42084432712488"


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def P(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------
# Exact univariate helpers over Fractions (for the real-root work)
# ---------------------------------------------------------------------------------------

def fr_poly(coeffs):
    return [Fraction(str(c)) for c in coeffs]


def fr_eval(p, x):
    acc = Fraction(0)
    for c in reversed(p):
        acc = acc * x + c
    return acc


def fr_deriv(p):
    return [k * p[k] for k in range(1, len(p))]


def fr_trim(p):
    while p and p[-1] == 0:
        p = p[:-1]
    return p


def fr_rem(a, b):
    """Remainder of a modulo b (lists of Fractions, low to high), b non-zero."""
    a = list(a)
    b = fr_trim(list(b))
    db = len(b) - 1
    while len(fr_trim(a)) - 1 >= db:
        a = fr_trim(a)
        da = len(a) - 1
        q = a[-1] / b[-1]
        for k in range(db + 1):
            a[da - db + k] -= q * b[k]
        a = a[:-1]
        if not a:
            break
    return fr_trim(a)


def sturm_sequence(p):
    seq = [fr_trim(p), fr_trim(fr_deriv(p))]
    while len(seq[-1]) > 1:
        r = fr_rem(seq[-2], seq[-1])
        if not r:
            break
        # scale by a positive constant to keep the numbers small (allowed in Sturm sequences)
        m = max(abs(c) for c in r)
        seq.append([-c / m for c in r])
    return seq


def sign_variations(seq, x):
    signs = []
    for p in seq:
        v = fr_eval(p, x)
        if v != 0:
            signs.append(1 if v > 0 else -1)
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def count_roots(seq, a, b):
    """Number of distinct real roots in (a, b] (Sturm), a < b, neither a root of seq[0]."""
    return sign_variations(seq, a) - sign_variations(seq, b)


def isolate_real_roots(p):
    """Isolating intervals (a, b] for the distinct real roots of p, Fractions, a < b."""
    p = fr_trim(p)
    seq = sturm_sequence(p)
    B = 1 + max(abs(c) for c in p[:-1]) / abs(p[-1])  # Cauchy bound
    B = Fraction(math.ceil(B))
    # make sure the endpoints are not roots
    while fr_eval(p, -B) == 0 or fr_eval(p, B) == 0:
        B += 1
    total = count_roots(seq, -B, B)
    work = [(-B, B)]
    out = []
    while work:
        a, b = work.pop()
        n = count_roots(seq, a, b)
        if n == 0:
            continue
        if n == 1:
            out.append((a, b))
            continue
        m = (a + b) / 2
        while fr_eval(p, m) == 0:
            m = (a + m) / 2
        work.append((a, m))
        work.append((m, b))
    out.sort()
    assert len(out) == total
    return out, total


def refine_root(p, a, b, bits):
    """Shrink (a, b] with p(a) p(b) < 0 by sign bisection to width <= (b - a) / 2^bits."""
    fa, fb = fr_eval(p, a), fr_eval(p, b)
    if not fa * fb < 0:
        # no sign change (a real root of even multiplicity): bisect on the Sturm count instead
        seq = sturm_sequence(p)
        for _ in range(bits):
            m = (a + b) / 2
            if fr_eval(p, m) == 0:
                return m, m
            if count_roots(seq, a, m) == 1:
                b = m
            else:
                a = m
        return a, b
    for _ in range(bits):
        m = (a + b) / 2
        fm = fr_eval(p, m)
        if fm == 0:
            return m, m
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b, fb = m, fm
    return a, b


def fr_to_arb(x):
    return arb(fmpq(x.numerator, x.denominator))


def interval_to_arb(a, b):
    return fr_to_arb(a).union(fr_to_arb(b))


def arb_poly_from_fr(p):
    return arb_poly([fr_to_arb(c) for c in p])


def arb_lower_bound(x):
    """Exact rational lower bound of an arb ball."""
    m, e = x.mid().man_exp()
    rm, re = x.rad().man_exp()
    mid = Fraction(int(m)) * (Fraction(2) ** int(e))
    rad = Fraction(int(rm)) * (Fraction(2) ** int(re))
    return mid - rad


def arb_upper_bound(x):
    m, e = x.mid().man_exp()
    rm, re = x.rad().man_exp()
    mid = Fraction(int(m)) * (Fraction(2) ** int(e))
    rad = Fraction(int(rm)) * (Fraction(2) ** int(re))
    return mid + rad


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------

def load(path):
    with open(path) as fh:
        d = json.load(fh)
    if d.get("complete") is not True:
        P(f"FILE NOT COMPLETE: {path} has complete = {d.get('complete')!r}; stopping")
        sys.exit(3)
    return d


def main():
    args = sys.argv[1:]
    path = "results/sigma_rur_inner1_tower.json"
    perturb = None
    skip_exact = False
    force_sign = None
    i = 0
    while i < len(args):
        if args[i] == "--perturb":
            perturb = args[i + 1]
            i += 2
        elif args[i] == "--skip-exact":
            skip_exact = True
            i += 1
        elif args[i] == "--sign":
            force_sign = int(args[i + 1])
            i += 2
        else:
            path = args[i]
            i += 1
    t_start = time.time()
    P("indep_exact_tower: start", stamp(), "file:", path)
    P("python-flint", flint.__version__)
    d = load(path)
    system = d["system"]
    tower = d["tower"]
    s = int(system["s"])
    p_order = int(system["p"])
    free = list(system["free"])
    P(f"system: s={s} p={p_order} fixed={len(system['fixed'])} free={len(free)}")
    P("file keys:", sorted(d.keys()), " tower keys:", sorted(tower.keys()))

    # ---- tree counts ----
    counts, total = core.check_tree_counts(p_order)
    P("tree counts by size:", counts, "total:", total)

    # ---- the ring K ----
    Fc = fr_poly(tower["F"])
    P0c = fr_poly(tower["P0"])
    P1c = fr_poly(tower["P1"])
    F = fmpq_poly([core.frac_to_fmpq(c) for c in Fc])
    P0 = fmpq_poly([core.frac_to_fmpq(c) for c in P0c])
    P1 = fmpq_poly([core.frac_to_fmpq(c) for c in P1c])
    degF = F.degree()
    P(f"F degree {degF}, P0 degree {P0.degree()}, P1 degree {P1.degree()}")
    P("F =", str(F)[:400])
    P("P1 =", str(P1)[:400])
    P("P0 =", str(P0)[:400])
    R = core.KRing(F, P0, P1)

    uv = tower["uv"]
    missing = [n for n in free if n not in uv]
    if missing:
        P("ERROR: free coordinates without a tower entry:", missing)
        sys.exit(4)
    extra_uv = [n for n in uv if n not in set(free)]
    P("uv entries not in free list (ignored):", extra_uv)
    max_u_deg = max(len(uv[n][0]) for n in free) - 1
    max_v_deg = max(len(uv[n][1]) for n in free) - 1
    max_digits = 0
    for n in free:
        for part in uv[n]:
            for c in part:
                max_digits = max(max_digits, len(str(c)))
    P(f"uv: max deg u {max_u_deg}, max deg v {max_v_deg}, longest coefficient string {max_digits} chars")

    def lookup(name):
        return R.from_strings(uv[name][0], uv[name][1])

    A, b, rep = core.build_tableau(system, lookup, R.from_rational)
    P("tableau built over K:", rep)
    n_v_nonzero = sum(1 for n in free if not lookup(n).v.is_zero())
    P(f"free coordinates with a non-zero y-part: {n_v_nonzero} of {len(free)}")

    # ---- irreducibility ----
    P("\n== irreducibility ==")
    F_fac = F.factor()
    P("F factorisation over Q (content, [(factor, mult)]): content =", F_fac[0],
      "factors:", [(str(f)[:80], e) for f, e in F_fac[1]], "degrees:", [f.degree() for f, e in F_fac[1]])
    F_irred = len(F_fac[1]) == 1 and F_fac[1][0][1] == 1 and F_fac[1][0][0].degree() == degF
    g = F.gcd(F.derivative())
    P("gcd(F, F') =", g, "-> F squarefree:", g.degree() == 0)
    P("F irreducible over Q:", F_irred)
    # multiplication-by-y matrix on the basis t^i (i < degF), t^i y (i < degF)
    n = 2 * degF
    rows = []
    for i in range(degF):
        # y * t^i = t^i y
        row = [fmpq(0)] * n
        row[degF + i] = fmpq(1)
        rows.append(row)
    for i in range(degF):
        # y * t^i y = t^i (P1 y + P0) = (t^i P0 mod F) + (t^i P1 mod F) y
        ti = fmpq_poly([0] * i + [1])
        u = (ti * P0) % F
        v = (ti * P1) % F
        row = [fmpq(0)] * n
        for k, c in enumerate(u.coeffs()):
            row[k] = c
        for k, c in enumerate(v.coeffs()):
            row[degF + k] = c
        rows.append(row)
    M = fmpq_mat(n, n, [rows[i][j] for i in range(n) for j in range(n)])
    Ry = M.charpoly()
    Ry_fac = Ry.factor()
    P("char poly of multiplication by y: degree", Ry.degree(), "factor degrees:",
      [(f.degree(), e) for f, e in Ry_fac[1]])
    y_irred = len(Ry_fac[1]) == 1 and Ry_fac[1][0][1] == 1 and Ry_fac[1][0][0].degree() == n
    P("char poly of y irreducible over Q of degree", n, ":", y_irred)
    # discriminant D = P1^2 + 4 P0 as a polynomial, and its norm from Q(t)
    Dpoly = P1 * P1 + 4 * P0
    Dred = Dpoly % F
    P("D = P1^2 + 4 P0 mod F:", str(Dred)[:300])
    if F_irred and y_irred:
        P("CONCLUSION: F irreducible of degree", degF, "and y has degree", n,
          "over Q, so y^2 - P1 y - P0 is irreducible over Q(t) and K is a field of degree", n)
    elif F_irred:
        # fallback: D is a square in Q(t) iff the quadratic splits; test by factoring z^2 - D over
        # the number field through the characteristic polynomial of multiplication by D
        P("y's char poly is reducible; the quadratic may still be irreducible (inconclusive here)")
    else:
        P("F is reducible; K is not a field, but the existence argument below does not need it")

    # ---- exact order conditions ----
    oc = core.OrderConditions(A, b, R.from_rational)
    if not skip_exact:
        P("\n== exact order conditions in K ==")
        t0 = time.time()

        def progress(k, nt, nz, secs):
            P(f"order {k}: {nt} trees, non-zero residuals: {nz}, {secs:.1f} s")

        res = oc.residuals_pruned(p_order, progress)
        summ = core.summarize_residuals(res)
        all_zero = True
        for k in sorted(summ):
            nt, nf, fl = summ[k]
            if nf:
                all_zero = False
                P(f"  FAIL order {k}: {nf} of {nt} trees non-zero: {fl[:20]}")
        P(f"exact check: {len(res)} conditions, all residuals exactly zero in K: {all_zero}, "
          f"{time.time() - t0:.1f} s")
        if all_zero:
            P("RESULT: every order condition up to order", p_order, "holds identically in K")
        else:
            P("RESULT: FAILURE, see the orders above")
            for tree, k, r in res:
                if not r.is_zero():
                    P("  first failing tree:", core.tree_str(tree), "order", k,
                      "u-part degree", r.u.degree(), "v-part degree", r.v.degree())
                    break

        # ---- perturbation control ----
        P("\n== perturbation control (u constant term + 1e-6 on one free coordinate) ==")
        pname = perturb or free[0]
        t0 = time.time()

        def lookup_p(name):
            e = lookup(name)
            if name == pname:
                e = e + R.from_rational(Fraction(1, 10 ** 6))
            return e

        Ap, bp, _ = core.build_tableau(system, lookup_p, R.from_rational)
        ocp = core.OrderConditions(Ap, bp, R.from_rational)
        resp = ocp.residuals_pruned(p_order, progress)
        summp = core.summarize_residuals(resp)
        fail_orders = [k for k in sorted(summp) if summp[k][1] > 0]
        for k in sorted(summp):
            nt, nf, fl = summp[k]
            P(f"  perturbed {pname}: order {k}: {nf} of {nt} conditions fail")
        P(f"perturbation control: failing orders {fail_orders}, "
          f"{sum(summp[k][1] for k in summp)} of {len(resp)} conditions fail, {time.time() - t0:.1f} s")
        P("perturbation control PASS (at least one condition fails):", len(fail_orders) > 0)

    # ---- real embedding ----
    P("\n== real embedding ==")
    t0 = time.time()
    roots, nreal = isolate_real_roots(Fc)
    P(f"F has {nreal} distinct real roots (Sturm); isolating intervals:")
    for a, bnd in roots:
        P(f"  ({float(a):.6f}, {float(bnd):.6f}]")
    BITS = 7000
    flint.ctx.prec = 8000
    lane_t0 = Fraction(LANE_T0)
    chosen = None
    embeddings = []
    for idx, (a, bnd) in enumerate(roots):
        lo, hi = refine_root(Fc, a, bnd, BITS)
        width = hi - lo
        # discriminant on [lo, hi]: exact bound  D(t) >= min(D(lo), D(hi)) - M (hi - lo)
        Dc = fr_poly([Fraction(int(c.p), int(c.q)) for c in Dpoly.coeffs()])
        Dlo, Dhi = fr_eval(Dc, lo), fr_eval(Dc, hi)
        tmax = max(abs(lo), abs(hi))
        Mbound = sum(k * abs(Dc[k]) * tmax ** (k - 1) for k in range(1, len(Dc)))
        Dmin_exact = min(Dlo, Dhi) - Mbound * width
        tball = interval_to_arb(lo, hi)
        Dball = arb_poly_from_fr(Dc)(tball)
        Dlow_arb = arb_lower_bound(Dball)
        pos = Dmin_exact > 0 and Dlow_arb > 0
        wlog2 = (width.numerator.bit_length() - width.denominator.bit_length()) if width else None
        P(f"root {idx}: t0 in [{float(lo):.18f}, {float(hi):.18f}], width about 2^{wlog2};"
          f" D(t0) exact lower bound {float(Dmin_exact):.6e}, arb enclosure {Dball.str(10, radius=True)},"
          f" D > 0 on the interval: {pos}")
        if pos:
            P1ball = arb_poly_from_fr(P1c)(tball)
            sq = Dball.sqrt()
            for sgn in (+1, -1):
                yball = (P1ball + sgn * sq) / 2
                embeddings.append((idx, sgn, lo, hi, tball, yball))
                P(f"    sign {sgn:+d}: y0 = {yball.str(25, radius=True)}")
        matches_t = lo - Fraction(1, 10 ** 15) <= lane_t0 <= hi + Fraction(1, 10 ** 15)
        if matches_t:
            chosen = idx
            P(f"    this root matches the lane's t0 = {LANE_T0} to the digits given")
    if chosen is None:
        P("No real root of F matches the lane's t0; using the first real root with D > 0 if any")
    cand = [e for e in embeddings if (chosen is None or e[0] == chosen)]
    if not cand:
        P("No real embedding with positive discriminant found; stopping here")
        sys.exit(5)
    lane_y0 = Fraction(LANE_Y0)
    pick = None
    for e in cand:
        ylo, yhi = arb_lower_bound(e[5]), arb_upper_bound(e[5])
        diff = min(abs(ylo - lane_y0), abs(yhi - lane_y0))
        P(f"sign {e[1]:+d}: |y0 - lane y0| about {float(diff):.3e}")
        if ylo - Fraction(1, 10 ** 8) <= lane_y0 <= yhi + Fraction(1, 10 ** 8):
            pick = e
            P(f"sign {e[1]:+d} matches the lane's y0 = {LANE_Y0} to about 1e-8 (nearest candidate)")
    if force_sign is not None:
        forced = [e for e in cand if e[1] == force_sign]
        if forced:
            pick = forced[0]
            P(f"sign {force_sign:+d} forced by --sign")
    if pick is None:
        pick = cand[0]
        P("no sign matches the lane's y0; using sign", pick[1], "of root", pick[0])
    idx, sgn, lo, hi, tball, yball = pick
    P(f"chosen embedding: real root {idx} of F (0-based, increasing order), sign {sgn:+d}")
    P("t0 =", tball.str(40, radius=True))
    P("y0 =", yball.str(40, radius=True))

    # tableau in arb at the embedding
    def eval_K(e):
        u = arb_poly([arb(c) for c in e.u.coeffs()]) if e.u.degree() >= 0 else None
        v = arb_poly([arb(c) for c in e.v.coeffs()]) if e.v.degree() >= 0 else None
        val = arb(0)
        if u is not None:
            val += u(tball)
        if v is not None:
            val += v(tball) * yball
        return val

    A_arb = [[eval_K(A[i][j]) for j in range(s)] for i in range(s)]
    b_arb = [eval_K(b[i]) for i in range(s)]
    c_arb = [sum(A_arb[i][j] for j in range(s)) for i in range(s)] if s else []
    P("nodes c_i (row sums) to 30 digits, with ball radius:")
    for i in range(s):
        P(f"  c{i + 1:2d} = {c_arb[i].str(30, radius=True)}")
    P("weights b_i to 30 digits, with ball radius:")
    for i in range(s):
        P(f"  b{i + 1:2d} = {b_arb[i].str(30, radius=True)}")
    P("sum of b_i =", sum(b_arb).str(30, radius=True))
    if s >= 12:
        P("consistency with the lane's naming: c3 - t0 =", (c_arb[2] - tball).str(5, radius=True),
          " c12 - y0 =", (c_arb[11] - yball).str(5, radius=True))
    P(f"real embedding done in {time.time() - t0:.1f} s")

    # ---- mpmath sanity check at 60 digits ----
    P("\n== mpmath residual check at 60 digits ==")
    import mpmath as mp
    t0 = time.time()
    mp.mp.dps = 2500
    tmid = (lo + hi) / 2
    t_mp = mp.mpf(tmid.numerator) / mp.mpf(tmid.denominator)

    def mp_poly(coeffs, x):
        acc = mp.mpf(0)
        for c in reversed(coeffs):
            acc = acc * x + mp.mpf(c.numerator) / mp.mpf(c.denominator)
        return acc

    Dc = fr_poly([Fraction(int(c.p), int(c.q)) for c in Dpoly.coeffs()])
    y_mp = (mp_poly(P1c, t_mp) + sgn * mp.sqrt(mp_poly(Dc, t_mp))) / 2

    def eval_K_mp(e):
        uc = [core.fmpq_to_frac(c) for c in e.u.coeffs()]
        vc = [core.fmpq_to_frac(c) for c in e.v.coeffs()]
        return mp_poly(uc, t_mp) + mp_poly(vc, t_mp) * y_mp

    A_hi = [[eval_K_mp(A[i][j]) for j in range(s)] for i in range(s)]
    b_hi = [eval_K_mp(b[i]) for i in range(s)]
    P("F(t0) at 2500 digits:", mp.nstr(mp_poly(Fc, t_mp), 5),
      " quadratic residual y0^2 - P1 y0 - P0:",
      mp.nstr(y_mp ** 2 - mp_poly(P1c, t_mp) * y_mp - mp_poly(P0c, t_mp), 5))
    for dps in (60, 100):
        mp.mp.dps = dps
        A_mp = [[+A_hi[i][j] for j in range(s)] for i in range(s)]
        b_mp = [+b_hi[i] for i in range(s)]

        def conv(x):
            x = Fraction(x)
            return mp.mpf(x.numerator) / mp.mpf(x.denominator)

        ocm = core.OrderConditions(A_mp, b_mp, conv)
        resm = ocm.residuals_up_to(p_order)
        worst = {}
        for tree, k, r in resm:
            worst[k] = max(worst.get(k, mp.mpf(0)), abs(r))
        P(f"mpmath dps={dps}: max |residual| by order:",
          {k: mp.nstr(worst[k], 3) for k in sorted(worst)},
          " overall max:", mp.nstr(max(worst.values()), 3))
    P(f"mpmath check done in {time.time() - t0:.1f} s")
    P(f"indep_exact_tower: done in {time.time() - t_start:.1f} s at {stamp()}")


if __name__ == "__main__":
    main()
