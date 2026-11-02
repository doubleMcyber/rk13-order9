"""Third exact check of the 13-stage order-9 certificate, in pure Python: CPython integers and fractions only.

  .venv/bin/python scripts/purepy_exact_check.py results/sigma_rur_inner1_tower.json [--max-order N] [--perturb NAME]

Why: the lane's verifier (scripts/fibre_verify.py) and the independent checker (scripts/indep_exact_*.py) both do
their exact arithmetic in python-flint, so FLINT was a shared point of failure. This script imports neither
python-flint nor sympy nor gmpy2 (it asserts so at the end) and shares no code with the lane: trees, elementary
weights and the field arithmetic are written here from Butcher's definitions.

Field: K = Q[t, y] / (F(t), y^2 - P1(t) y - P0(t)) from the file's "tower" key. An element is (u, v) with
u, v in Q[t]/(F), each stored as eight integers over one positive common denominator, reduced by their gcd.
Order conditions: for every rooted tree T with at most p vertices, sum_i b_i Phi_i(T) = 1/gamma(T), where
Phi_i(leaf) = 1, Phi_i([T1, ..., Tk]) = prod_m (sum_j a_ij Phi_j(T_m)), gamma(T) = |T| prod_m gamma(T_m).
A condition holds when its residual is exactly zero in K (both parts zero).
Real embedding: a real root t0 of F isolated by Sturm counts in exact fractions, and P1(t0)^2 + 4 P0(t0) > 0 shown
by exact endpoint values and a derivative bound on the interval, so y0 is real and evaluation at (t0, y0) is a
ring homomorphism from K to R.
Controls: tree counts against A000081; classical RK4 (orders 1-4 hold, order 5 fails); the data with
1/10^6 added to the constant term of one coordinate's u must fail.
"""
import json
import math
import sys
import time
from fractions import Fraction

DEG = 8  # degree of F, checked on load


# ---------------------------------------------------------------------------------------------------------------
# Q[t]/(F): eight integers over a common denominator
class Base:
    F = None  # integer coefficients, low degree first
    R = None  # R[k - DEG] = integer vector for t^k, k = DEG .. 2 DEG - 2, all over the common denominator RDEN
    RDEN = 1

    @classmethod
    def setup(cls, F):
        assert len(F) == DEG + 1 and F[-1] != 0
        cls.F = F
        L = F[-1]
        # t^DEG = -(1/L) sum_{k<DEG} F_k t^k, then t^(k+1) = t * t^k reduced, in exact fractions
        red = [[Fraction(-F[k], L) for k in range(DEG)]]
        for _ in range(DEG - 2):
            prev = red[-1]
            top = prev[DEG - 1]
            nxt = [Fraction(0)] + prev[:DEG - 1]
            nxt = [nxt[k] + top * red[0][k] for k in range(DEG)]
            red.append(nxt)
        den = 1
        for row in red:
            for c in row:
                den = den * c.denominator // math.gcd(den, c.denominator)
        cls.RDEN = den
        cls.R = [[int(c * den) for c in row] for row in red]

    def __init__(self, ints, den=1, normalize=True):
        self.c = ints
        self.d = den
        if normalize:
            self._norm()

    def _norm(self):
        if self.d < 0:
            self.d = -self.d
            self.c = [-x for x in self.c]
        if not any(self.c):
            self.d = 1
            return
        g = self.d
        for x in self.c:
            if x:
                g = math.gcd(g, x)
                if g == 1:
                    return
        if g > 1:
            self.d //= g
            self.c = [x // g for x in self.c]

    @classmethod
    def from_fracs(cls, fr):
        fr = [Fraction(x) for x in fr] + [Fraction(0)] * (DEG - len(fr))
        assert len(fr) == DEG, "polynomial of degree >= DEG in the data"
        den = 1
        for x in fr:
            den = den * x.denominator // math.gcd(den, x.denominator)
        return cls([int(x * den) for x in fr], den)

    @classmethod
    def const(cls, q):
        q = Fraction(q)
        return cls([q.numerator] + [0] * (DEG - 1), q.denominator)

    def is_zero(self):
        return not any(self.c)

    def __add__(self, o):
        if self.is_zero():
            return o
        if o.is_zero():
            return self
        if self.d == o.d:
            return Base([a + b for a, b in zip(self.c, o.c)], self.d)
        g = math.gcd(self.d, o.d)
        l = self.d // g * o.d
        fa, fb = l // self.d, l // o.d
        return Base([a * fa + b * fb for a, b in zip(self.c, o.c)], l)

    def __neg__(self):
        return Base([-a for a in self.c], self.d, normalize=False)

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        if self.is_zero() or o.is_zero():
            return Base([0] * DEG)
        a, b = self.c, o.c
        conv = [0] * (2 * DEG - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    if y:
                        conv[i + j] += x * y
        out = [conv[k] * Base.RDEN for k in range(DEG)]
        for k in range(DEG, 2 * DEG - 1):
            ck = conv[k]
            if ck:
                row = Base.R[k - DEG]
                for m in range(DEG):
                    out[m] += ck * row[m]
        return Base(out, self.d * o.d * Base.RDEN)

    def to_fracs(self):
        return [Fraction(x, self.d) for x in self.c]


# ---------------------------------------------------------------------------------------------------------------
# K = Base[y]/(y^2 - P1 y - P0)
class K:
    P0 = None
    P1 = None

    def __init__(self, u, v=None):
        self.u = u
        self.v = v if v is not None else Base([0] * DEG)

    @classmethod
    def const(cls, q):
        return cls(Base.const(q))

    def is_zero(self):
        return self.u.is_zero() and self.v.is_zero()

    def __add__(self, o):
        return K(self.u + o.u, self.v + o.v)

    def __sub__(self, o):
        return K(self.u - o.u, self.v - o.v)

    def __mul__(self, o):
        if self.v.is_zero() and o.v.is_zero():
            return K(self.u * o.u)
        if self.v.is_zero():
            return K(self.u * o.u, self.u * o.v)
        if o.v.is_zero():
            return K(self.u * o.u, self.v * o.u)
        vv = self.v * o.v
        u = self.u * o.u + vv * K.P0
        v = self.u * o.v + self.v * o.u + vv * K.P1
        return K(u, v)


# ---------------------------------------------------------------------------------------------------------------
# rooted trees: a tree is the sorted tuple of its children's trees; the leaf is ()
def trees_up_to(n):
    by_size = {1: [()]}
    for size in range(2, n + 1):
        found = set()
        # children multisets with sizes summing to size - 1
        def build(rem, max_key, acc):
            if rem == 0:
                found.add(tuple(sorted(acc)))
                return
            for s in range(1, rem + 1):
                for t in by_size[s]:
                    key = (s, t)
                    if max_key is not None and key > max_key:
                        continue
                    build(rem - s, key, acc + [t])
        build(size - 1, None, [])
        by_size[size] = sorted(found)
    return by_size


def tree_size(t):
    return 1 + sum(tree_size(c) for c in t)


def gamma(t):
    g = tree_size(t)
    for c in t:
        g *= gamma(c)
    return g


# ---------------------------------------------------------------------------------------------------------------
def check(A, b, s, p, by_size, label, verbose=True):
    """Exact residuals of all trees with at most p vertices. Returns {order: [nonzero count, total]}."""
    one = K.const(1)
    zero = K.const(0)
    phi = {(): [one] * s}
    aphi = {}

    def get_aphi(t):
        if t not in aphi:
            ph = get_phi(t)
            vec = []
            for i in range(s):
                acc = zero
                for j in range(i):
                    if A[i][j] is not None and not ph[j].is_zero():
                        acc = acc + A[i][j] * ph[j]
                vec.append(acc)
            aphi[t] = vec
        return aphi[t]

    def get_phi(t):
        if t not in phi:
            vec = None
            for c in t:
                ac = get_aphi(c)
                vec = list(ac) if vec is None else [vec[i] * ac[i] for i in range(s)]
            phi[t] = vec
        return phi[t]

    out = {}
    for n in range(1, p + 1):
        t0 = time.time()
        bad = 0
        for t in by_size[n]:
            ph = get_phi(t)
            acc = zero
            for i in range(s):
                if b[i] is not None and not ph[i].is_zero():
                    acc = acc + b[i] * ph[i]
            res = acc - K.const(Fraction(1, gamma(t)))
            if not res.is_zero():
                bad += 1
            if n == p:
                del phi[t]  # no larger tree uses it; keeps memory down
        out[n] = [bad, len(by_size[n])]
        if verbose:
            print(f"  [{label}] order {n}: {len(by_size[n])} trees, nonzero residuals {bad} ({time.time() - t0:.1f} s)", flush=True)
    return out


def sturm_count(seq, x):
    signs = []
    for q in seq:
        v = sum(c * x ** k for k, c in enumerate(q))
        if v != 0:
            signs.append(v > 0)
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def poly_rem(a, b):
    a = list(a)
    while len(a) >= len(b) and any(a):
        if a[-1] == 0:
            a.pop()
            continue
        f = a[-1] / b[-1]
        sh = len(a) - len(b)
        for k in range(len(b)):
            a[sh + k] -= f * b[k]
        a.pop()
    while a and a[-1] == 0:
        a.pop()
    return a


def main():
    args = sys.argv[1:]
    path = args[0]
    max_order = int(args[args.index("--max-order") + 1]) if "--max-order" in args else None
    t_start = time.time()
    d = json.load(open(path))
    tw = d["tower"]
    F = [int(x) for x in tw["F"]]
    assert len(F) == DEG + 1
    Base.setup(F)
    K.P0 = Base.from_fracs(tw["P0"])
    K.P1 = Base.from_fracs(tw["P1"])
    sysd = d["system"]
    s, p = sysd["s"], sysd["p"]
    if max_order:
        p = min(p, max_order)
    fixed = {k: Fraction(v) for k, v in sysd["fixed"].items()}
    uv = tw["uv"]
    names = [f"a{i},{j}" for i in range(2, s + 1) for j in range(1, i)] + [f"b{i}" for i in range(1, s + 1)]
    assert set(fixed) | set(uv) == set(names) and not (set(fixed) & set(uv)), "fixed and free do not partition the tableau"
    print(f"data: s={s}, p={p}, {len(fixed)} fixed, {len(uv)} free; F degree {len(F) - 1}, leading coefficient {len(str(F[-1]))} digits; "
          f"reduction denominator {len(str(Base.RDEN))} digits", flush=True)

    def build(perturb=None):
        A = [[None] * s for _ in range(s)]
        b = [None] * s
        for nm in names:
            if nm in fixed:
                val = None if fixed[nm] == 0 else K.const(fixed[nm])
            else:
                u, v = uv[nm]
                uf = [Fraction(x) for x in u]
                if nm == perturb:
                    uf = (uf or [Fraction(0)])
                    uf[0] += Fraction(1, 10 ** 6)
                val = K(Base.from_fracs(uf), Base.from_fracs(v))
            if nm.startswith("a"):
                i, j = (int(x) for x in nm[1:].split(","))
                A[i - 1][j - 1] = val
            else:
                b[int(nm[1:]) - 1] = val
        return A, b

    by_size = trees_up_to(p)
    counts = [len(by_size[n]) for n in range(1, p + 1)]
    print(f"tree counts by size: {counts}, total {sum(counts)} (A000081: [1, 1, 2, 4, 9, 20, 48, 115, 286][:{p}])", flush=True)
    assert counts == [1, 1, 2, 4, 9, 20, 48, 115, 286][:p]

    # control 1: classical RK4, embedded in K as constants
    h = Fraction(1, 2)
    A4 = [[None] * 4 for _ in range(4)]
    A4[1][0], A4[2][1], A4[3][2] = K.const(h), K.const(h), K.const(1)
    b4 = [K.const(Fraction(1, 6)), K.const(Fraction(1, 3)), K.const(Fraction(1, 3)), K.const(Fraction(1, 6))]
    r4 = check(A4, b4, 4, 5, trees_up_to(5), "RK4")
    ok4 = all(r4[n][0] == 0 for n in range(1, 5)) and r4[5][0] > 0
    print(f"control RK4: orders 1-4 all zero, {r4[5][0]} of 9 order-5 trees nonzero: {'as expected' if ok4 else 'FAILED'}", flush=True)

    # the certificate
    t0 = time.time()
    A, b = build()
    res = check(A, b, s, p, by_size, "data")
    all_zero = all(v[0] == 0 for v in res.values())
    n_total = sum(v[1] for v in res.values())
    print(f"RESULT: {n_total} conditions, all exactly zero in K: {all_zero} ({time.time() - t0:.1f} s)", flush=True)

    # control 2: perturbation
    pert = "a3,1"
    A, b = build(perturb=pert)
    rp = check(A, b, s, p, by_size, f"perturbed {pert}", verbose=False)
    nbad = sum(v[0] for v in rp.values())
    print(f"control perturbation ({pert} + 1/10^6 in u's constant term): failing by order {[(n, v[0]) for n, v in rp.items() if v[0]]}, {nbad} of {n_total} fail", flush=True)

    # real embedding: every real root of F isolated by Sturm counts in exact fractions; at each, the sign of
    # D = P1^2 + 4 P0 on the isolating interval from exact endpoint values and a bound on |D'|. Zero residuals in
    # the ring Q[t, y]/(F, y^2 - P1 y - P0) map to zero under t -> t0, y -> y0 for any real t0 with D(t0) > 0 and
    # either real root y0, so each such root gives two real methods. No division is used above, so this needs no
    # field property.
    Ff = [Fraction(x) for x in F]
    seq = [Ff, [k * Ff[k] for k in range(1, len(Ff))]]
    while len(seq[-1]) > 1:
        r = poly_rem(seq[-2], seq[-1])
        if not r:
            break
        seq.append([-x for x in r])
    squarefree = len(seq[-1]) == 1
    bound = 1 + max(abs(x) for x in Ff[:-1]) / abs(Ff[-1])  # Cauchy bound on the roots
    n_real = sturm_count(seq, -bound) - sturm_count(seq, bound)
    print(f"F: squarefree {squarefree}; {n_real} distinct real roots in (-{float(bound):.3g}, {float(bound):.3g}] by Sturm (exact fractions)", flush=True)
    boxes = []

    def isolate(lo, hi, vlo, vhi):
        n = vlo - vhi
        if n == 0:
            return
        if n == 1:
            boxes.append([lo, hi])
            return
        mid = (lo + hi) / 2
        vm = sturm_count(seq, mid)
        isolate(lo, mid, vlo, vm)
        isolate(mid, hi, vm, vhi)

    isolate(-bound, bound, sturm_count(seq, -bound), sturm_count(seq, bound))
    assert len(boxes) == n_real
    P0f, P1f = K.P0.to_fracs(), K.P1.to_fracs()
    Dpoly = [Fraction(0)] * (2 * DEG)
    for i, x in enumerate(P1f):
        for j, y in enumerate(P1f):
            Dpoly[i + j] += x * y
    for i, x in enumerate(P0f):
        Dpoly[i] += 4 * x

    def ev(q, x):
        return sum(c * x ** k for k, c in enumerate(q))

    n_pos = 0
    for lo, hi in boxes:
        shown = None
        for _ in range(400):
            M = max(abs(lo), abs(hi), Fraction(1))
            dbound = sum(abs(c) * k * M ** (k - 1) for k, c in enumerate(Dpoly) if k)
            dlo, dhi = ev(Dpoly, lo), ev(Dpoly, hi)
            if min(dlo, dhi) - dbound * (hi - lo) > 0:
                shown = "D > 0"
                break
            if max(dlo, dhi) + dbound * (hi - lo) < 0:
                shown = "D < 0"
                break
            mid = (lo + hi) / 2
            if sturm_count(seq, lo) - sturm_count(seq, mid) == 1:
                hi = mid
            else:
                lo = mid
        n_pos += shown == "D > 0"
        print(f"  root t0 in [{float(lo):.17g}, {float(hi):.17g}], width {float(hi - lo):.2g}: {shown or 'sign NOT decided'}", flush=True)
    print(f"real roots of F with D > 0: {n_pos} of {n_real}, so {2 * n_pos} real methods", flush=True)
    for mod in ("flint", "sympy", "gmpy2", "mpmath", "numpy"):
        assert mod not in sys.modules, f"{mod} was imported"
    print(f"modules: no flint, sympy, gmpy2, mpmath or numpy loaded. Python {sys.version.split()[0]}", flush=True)
    verdict = all_zero and ok4 and nbad > 0 and n_pos >= 1
    print(f"VERDICT: {'CERTIFIED by pure Python' if verdict else 'NOT CERTIFIED'}; total {time.time() - t_start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
