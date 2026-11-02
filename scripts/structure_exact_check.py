"""Exact check, in the certified field K, of the structural properties the paper states for the inner1 tableau.

  .venv/bin/python scripts/structure_exact_check.py results/sigma_rur_inner1_tower.json

Uses the pure-Python ring arithmetic of scripts/purepy_exact_check.py (K = Q[t, y]/(F, y^2 - P1 y - P0)). Not part of
the proof; it turns observations made numerically on Sigma into exact statements about the certified data:
  stage order: the largest eta <= 6 with sum_j a_ij c_j^(k-1) = c_i^k / k for k = 1..eta, at every stage i;
  D(zeta):     the largest zeta <= 9 with sum_i b_i c_i^(k-1) a_ij = b_j (1 - c_j^k) / k for k = 1..zeta, at column j;
  coincidences of nodes and of the column-2 entries, and the identity a3,2 = c3^2 / (2 c2).
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

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
    b = [zero] * s
    for nm in [f"a{i},{j}" for i in range(2, s + 1) for j in range(1, i)] + [f"b{i}" for i in range(1, s + 1)]:
        if nm in fixed:
            val = K.const(fixed[nm])
        else:
            u, v = uv[nm]
            val = K(pp.Base.from_fracs([Fraction(x) for x in u]), pp.Base.from_fracs([Fraction(x) for x in v]))
        if nm.startswith("a"):
            i, j = (int(x) for x in nm[1:].split(","))
            A[i - 1][j - 1] = val
        else:
            b[int(nm[1:]) - 1] = val

    def ksum(xs):
        out = zero
        for x in xs:
            out = out + x
        return out

    def kpow(x, k):
        out = K.const(Fraction(1))
        for _ in range(k):
            out = out * x
        return out

    c = [ksum(A[i]) for i in range(s)]
    one = K.const(Fraction(1))

    print("stage order (largest eta <= 6 with C(1..eta) at stage i):")
    for i in range(s):
        eta = 0
        for k in range(1, 7):
            lhs = ksum(A[i][j] * kpow(c[j], k - 1) for j in range(i))
            if not (lhs - kpow(c[i], k) * K.const(Fraction(1, k))).is_zero():
                break
            eta = k
        print(f"  stage {i + 1:2d}: b {'zero' if b[i].is_zero() else 'nonzero'}, C({eta})", flush=True)

    print("D (largest zeta <= 9 with D(1..zeta) at column j):")
    for j in range(s):
        zeta = 0
        for k in range(1, 10):
            lhs = ksum(b[i] * kpow(c[i], k - 1) * A[i][j] for i in range(j + 1, s))
            rhs = b[j] * (one - kpow(c[j], k)) * K.const(Fraction(1, k))
            if not (lhs - rhs).is_zero():
                break
            zeta = k
        print(f"  column {j + 1:2d}: D({zeta})", flush=True)

    print("coincidences:")
    print(f"  c4 = c9: {(c[3] - c[8]).is_zero()}; c4 = c10: {(c[3] - c[9]).is_zero()}; c13 = 1: {(c[12] - one).is_zero()}")
    print(f"  a9,2 = a4,2: {(A[8][1] - A[3][1]).is_zero()}; a9,2 nonzero: {not A[8][1].is_zero()}")
    print(f"  a3,2 = c3^2 / (2 c2): {(A[2][1] * K.const(2 * fixed['a2,1']) - c[2] * c[2]).is_zero()}")
    print(f"  stored a4,2 and a9,2 coordinates identical: {uv['a4,2'] == uv['a9,2']}")
    nz = [i + 1 for i in range(s) if not b[i].is_zero()]
    print(f"  stages with nonzero weight: {nz}")
    pairs = [(nz[m], nz[n]) for m in range(len(nz)) for n in range(m + 1, len(nz)) if (c[nz[m] - 1] - c[nz[n] - 1]).is_zero()]
    print(f"  equal nodes among weighted stages: {pairs if pairs else 'none'}")


if __name__ == "__main__":
    main()
