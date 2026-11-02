"""Exact test of whether an algebraic number beta (minimal polynomial g over Q) lies in Q(alpha) (minimal polynomial f).

  .venv/bin/python scripts/subfield_test.py MINPOLY_JSON ALPHA BETA [--k 1 2 3]

MINPOLY_JSON is a results/sigma_chart_*_minpoly*.json file; ALPHA and BETA are coordinate names in it (for example c3
and c12). Trager's norm: N_k(z) = prod over roots (alpha_i, beta_j) of (z - k alpha_i - beta_j), the composed sum of
k*alpha and beta, computed exactly from power sums (Newton's identities over Fractions), then factored over Q in FLINT.
When N_k is squarefree, its irreducible factors of degree deg(f) * e correspond one to one to the irreducible factors
of degree e of g over Q(alpha). So beta lies in Q(alpha) exactly when g has a linear factor there, that is when N_k
has a factor of degree deg(f). The minimal polynomials themselves are LLL relations at 3990 digits (numerical
evidence); given them, the test is exact. Post-certificate exploration (notes/simpler_method_plan.md).
"""
import argparse
import json
from fractions import Fraction

from flint import fmpq, fmpq_poly


def power_sums(c, m):
    """p_1..p_m of the roots of sum c[k] x^k (low degree first), by Newton's identities."""
    n = len(c) - 1
    e = [Fraction(c[n - k], c[n]) * (-1) ** k for k in range(n + 1)]  # elementary symmetric e_0..e_n
    p = [Fraction(n)]
    for k in range(1, m + 1):
        s = Fraction(0)
        for i in range(1, min(k, n) + 1):
            s += (-1) ** (i - 1) * e[i] * (p[k - i] if k - i > 0 else 0)
        if k <= n:
            s += (-1) ** (k - 1) * k * e[k]
        p.append(s)
    return p


def from_power_sums(p, n):
    """Monic polynomial of degree n (low degree first) whose roots have power sums p[1..n]."""
    e = [Fraction(1)]
    for k in range(1, n + 1):
        s = sum((-1) ** (i - 1) * e[k - i] * p[i] for i in range(1, k + 1))
        e.append(s / k)
    return [(-1) ** (n - k) * e[n - k] for k in range(n + 1)]


def composed_sum(f, g, k):
    n, m = len(f) - 1, len(g) - 1
    N = n * m
    pf, pg = power_sums(f, N), power_sums(g, N)
    binom = [[1]]
    for r in range(1, N + 1):
        binom.append([1] + [binom[-1][i - 1] + binom[-1][i] for i in range(1, r)] + [1])
    ps = [Fraction(N)]
    for r in range(1, N + 1):
        ps.append(sum(binom[r][l] * Fraction(k) ** l * pf[l] * pg[r - l] for l in range(r + 1)))
    return from_power_sums(ps, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("minpoly_json")
    ap.add_argument("alpha")
    ap.add_argument("beta")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    r = json.load(open(a.minpoly_json))["results"]
    f = [int(x) for x in r[a.alpha]["found"]["coeffs"]]
    g = [int(x) for x in r[a.beta]["found"]["coeffs"]]
    for poly, nm in ((f, a.alpha), (g, a.beta)):
        _, fac = fmpq_poly(poly).factor()
        assert len(fac) == 1 and fac[0][1] == 1, f"{nm}: minimal polynomial not irreducible: {[(q.degree(), e) for q, e in fac]}"
    df = len(f) - 1
    for k in a.k:
        N = composed_sum(f, g, k)
        P = fmpq_poly([fmpq(x.numerator, x.denominator) for x in N])
        sqfree = P.gcd(P.derivative()).degree() == 0
        _, fac = P.factor()
        degs = sorted(q.degree() for q, _ in fac)
        if not sqfree:
            print(f"k = {k}: N_k not squarefree, factor degrees {degs}; try another k")
            continue
        over = sorted(d // df for d in degs)
        verdict = "IN" if 1 in over else "NOT IN"
        print(f"k = {k}: N_k squarefree of degree {P.degree()}, factor degrees {degs}; so {a.beta}'s minimal polynomial "
              f"factors over Q({a.alpha}) with degrees {over}: {a.beta} {verdict} Q({a.alpha})")
        return
    print("no squarefree N_k for the k tried")


if __name__ == "__main__":
    main()
