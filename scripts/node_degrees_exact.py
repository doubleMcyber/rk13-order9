"""Exact degrees over Q of the 13 nodes c_i = sum_j a_ij of the certified tableau, in K = Q[t, y]/(F, y^2 - P1 y - P0).

  .venv/bin/python scripts/node_degrees_exact.py results/sigma_rur_inner1_tower.json

For each node, the 16 x 16 rational matrix of multiplication by c_i on the basis t^k y^j (k <= 7, j <= 1) is built with
the ring arithmetic of scripts/purepy_exact_check.py, its characteristic polynomial is computed and factored over Q in
FLINT, and the degree of the minimal polynomial is read off (the characteristic polynomial is a power of it, since K is
a field). It also records whether the node lies in Q(t) = Q(c3) (v = 0 in the tower form) and writes the exact
minimal polynomials as primitive integer polynomials, for scripts/subfield_test.py. Not part of the proof; it confirms the node degrees of the paper's Table 2 from the exact data (referee
report independent/REVIEW_paper_draft1.md, item B3).
"""
import json
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path

from flint import fmpq, fmpq_mat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import purepy_exact_check as pp  # noqa: E402


def main():
    d = json.load(open(sys.argv[1]))
    tw = d["tower"]
    pp.Base.setup([int(x) for x in tw["F"]])
    K = pp.K
    K.P0 = pp.Base.from_fracs(tw["P0"])
    K.P1 = pp.Base.from_fracs(tw["P1"])
    s = d["system"]["s"]
    fixed = {k: Fraction(v) for k, v in d["system"]["fixed"].items()}
    uv = tw["uv"]
    zero = K.const(Fraction(0))
    A = [[zero] * s for _ in range(s)]
    for i in range(2, s + 1):
        for j in range(1, i):
            nm = f"a{i},{j}"
            if nm in fixed:
                A[i - 1][j - 1] = K.const(fixed[nm])
            else:
                u, v = uv[nm]
                A[i - 1][j - 1] = K(pp.Base.from_fracs([Fraction(x) for x in u]), pp.Base.from_fracs([Fraction(x) for x in v]))
    deg = pp.DEG
    basis = []
    for j in range(2):
        for k in range(deg):
            e = [Fraction(0)] * deg
            e[k] = Fraction(1)
            eb = pp.Base.from_fracs(e)
            basis.append(K(eb) if j == 0 else K(pp.Base.from_fracs([Fraction(0)] * deg), eb))
    out = {}
    for i in range(s):
        c = zero
        for x in A[i]:
            c = c + x
        cols = []
        for e in basis:
            p = c * e
            cols.append(list(p.u.to_fracs()) + list(p.v.to_fracs()))
        n = 2 * deg
        M = fmpq_mat(n, n, [fmpq(cols[col][row].numerator, cols[col][row].denominator) for row in range(n) for col in range(n)])
        cp = M.charpoly()
        _, facs = cp.factor()
        mdeg = [(f.degree(), e) for f, e in facs]
        assert len(mdeg) == 1 and mdeg[0][0] * mdeg[0][1] == n, mdeg
        mp = [Fraction(int(q.p), int(q.q)) for q in facs[0][0].coeffs()]  # low degree first
        den = 1
        for q in mp:
            den = den * q.denominator // gcd(den, q.denominator)
        ints = [int(q * den) for q in mp]
        g = 0
        for z in ints:
            g = gcd(g, z)
        ints = [z // g for z in ints]
        out[f"c{i + 1}"] = {"degree": mdeg[0][0], "charpoly_power": mdeg[0][1],
                            "max_digits": max(len(str(abs(z))) for z in ints),
                            "in_Q_t": c.v.is_zero(), "found": {"degree": mdeg[0][0], "coeffs": [str(z) for z in ints]}}
        print(f"c{i + 1}: minimal polynomial degree {mdeg[0][0]} (characteristic polynomial = its power {mdeg[0][1]}), "
              f"integer coefficients of up to {out[f'c{i + 1}']['max_digits']} digits, in Q(t) = Q(c3): {c.v.is_zero()}", flush=True)
    # "results" layout as in the sigma_chart_*_minpoly JSON files, so scripts/subfield_test.py reads it
    Path("results/node_degrees_exact.json").write_text(json.dumps({"source": sys.argv[1], "results": out}, indent=1) + "\n")


if __name__ == "__main__":
    main()
