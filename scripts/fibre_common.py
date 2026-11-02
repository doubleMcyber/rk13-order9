"""Shared pieces of the fibre-to-certificate pipeline (lane K, PREREG line 5, exact-point route).

A fibre is the finite set of points of one component of the Runge-Kutta solution variety over fixed rational
values of a coordinate chart. The pipeline (scripts/fibre_refine.py, fibre_rur.py, fibre_verify.py,
fibre_trace.py) turns a numerically complete fibre into an exact certificate: a univariate polynomial f in Q[T]
and polynomials g_j in Q[T] such that every coordinate is x_j = g_j(t) / f'(t) at every root t of f, and all
order conditions hold modulo f.

This module holds the system description (an explicit s-stage method with the order-p tree conditions, some
coordinates fixed at rationals) and its evaluators:
  residual_np / jacobian_np      complex double (numpy), for path tracking
  residual_acb / jacobian_acb    python-flint acb at a given precision, for refinement
  verify_exact                   fmpq_poly arithmetic modulo f, for the certificate
plus rational reconstruction and JSON helpers. Coordinates are named as in scripts/monodromy_eval.py:
"a{i},{j}" (1-based, j < i) and "b{i}". Nodes are c_i = sum_j a_ij as in scripts/order_conditions.py, and the
tree order is that of order_conditions.py (trees(1), trees(2), ...).
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import order_conditions as oc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def prec_bits(dps):
    return int(dps * 3.33) + 64


class RKSystem:
    """Explicit s-stage Runge-Kutta method, order-p tree conditions, with the coordinates in `fixed` held at
    exact rationals. The unknown vector x lists the free coordinates in `self.free` order."""

    def __init__(self, s, p, fixed):
        self.s, self.p = s, p
        self.names = [f"a{i + 1},{j + 1}" for i in range(1, s) for j in range(i)] + [f"b{i + 1}" for i in range(s)]
        self.fixed = {n: Fraction(v) for n, v in fixed.items()}
        unknown = [n for n in self.fixed if n not in self.names]
        if unknown:
            raise ValueError(f"fixed coordinates not in the system: {unknown}")
        self.free = [n for n in self.names if n not in self.fixed]
        self.n = len(self.free)
        self.trees = [t for n in range(1, p + 1) for t in oc.trees(n)]
        self.tree_order = [n for n in range(1, p + 1) for _ in oc.trees(n)]
        self.inv_gamma = [Fraction(1, oc.gamma(t)) for t in self.trees]
        self.m = len(self.trees)
        self.avars = [self._parse(n) for n in self.free if n.startswith("a")]
        self.bvars = [self._parse(n) for n in self.free if n.startswith("b")]
        self.K = len(self.avars)
        self.arow = np.array([i for i, _ in self.avars], dtype=int)
        self.acol = np.array([j for _, j in self.avars], dtype=int)
        self.A_fixed = np.zeros((s, s), dtype=complex)
        self.b_fixed = np.zeros(s, dtype=complex)
        for nm, v in self.fixed.items():
            idx = self._parse(nm)
            if nm.startswith("a"):
                self.A_fixed[idx] = float(v)
            else:
                self.b_fixed[idx] = float(v)

    @staticmethod
    def _parse(nm):
        if nm.startswith("a"):
            i, j = nm[1:].split(",")
            return int(i) - 1, int(j) - 1
        return int(nm[1:]) - 1

    def to_json(self):
        return {"s": self.s, "p": self.p, "fixed": {k: str(v) for k, v in self.fixed.items()},
                "free": self.free, "n_conditions": self.m}

    @classmethod
    def from_json(cls, d):
        return cls(d["s"], d["p"], d["fixed"])

    # ---- assembling tableaux -------------------------------------------------------------------------------
    def tableau_np(self, x):
        A = self.A_fixed.copy()
        b = self.b_fixed.copy()
        A[self.arow, self.acol] = x[: self.K]
        for k, i in enumerate(self.bvars):
            b[i] = x[self.K + k]
        return A, b

    def tableau_generic(self, x, zero, conv):
        """A, b as lists with entries conv(Fraction) for fixed coordinates and x[k] for free ones."""
        A = [[zero] * self.s for _ in range(self.s)]
        b = [zero] * self.s
        for nm, v in self.fixed.items():
            idx = self._parse(nm)
            if nm.startswith("a"):
                A[idx[0]][idx[1]] = conv(v)
            else:
                b[idx] = conv(v)
        for k, (i, j) in enumerate(self.avars):
            A[i][j] = x[k]
        for k, i in enumerate(self.bvars):
            b[i] = x[self.K + k]
        return A, b

    # ---- numpy evaluators ------------------------------------------------------------------------------------
    def residual_np(self, x, want_jac=False):
        """Returns (r, ra) or (r, J, ra): r the tree residuals b.u(t) - 1/gamma(t), ra the size of the
        cancelling terms (roundoff floor = eps * ra), J the Jacobian with respect to the free coordinates."""
        s, K = self.s, self.K
        A, b = self.tableau_np(x)
        memo = {}

        def u(t):
            if t in memo:
                return memo[t]
            if not t:
                val = (np.ones(s, dtype=complex), np.zeros((s, K), dtype=complex) if want_jac else None)
            else:
                ws, dws = [], []
                for c in t:
                    uc, duc = u(c)
                    ws.append(A @ uc)
                    if want_jac:
                        dA = np.zeros((s, K), dtype=complex)
                        dA[self.arow, np.arange(K)] = uc[self.acol]
                        dws.append(dA + A @ duc)
                uval = ws[0].copy()
                for w in ws[1:]:
                    uval = uval * w
                du = None
                if want_jac:
                    du = np.zeros((s, K), dtype=complex)
                    for k in range(len(ws)):
                        others = np.ones(s, dtype=complex)
                        for l in range(len(ws)):
                            if l != k:
                                others = others * ws[l]
                        du += others[:, None] * dws[k]
                val = (uval, du)
            memo[t] = val
            return val

        U = np.empty((self.m, s), dtype=complex)
        J = np.empty((self.m, self.n), dtype=complex) if want_jac else None
        for row, t in enumerate(self.trees):
            ut, dut = u(t)
            U[row] = ut
            if want_jac:
                J[row, :K] = b @ dut
                J[row, K:] = ut[self.bvars]
        ig = np.array([float(v) for v in self.inv_gamma])
        r = U @ b - ig
        ra = np.abs(U) @ np.abs(b) + ig
        return (r, J, ra) if want_jac else (r, ra)

    def jacobian_np(self, x):
        return self.residual_np(x, want_jac=True)

    # ---- acb evaluators --------------------------------------------------------------------------------------
    def residual_acb(self, x, prec):
        """x: list of acb (free coordinates). Returns the list of tree residuals as acb at `prec` bits."""
        import flint
        from flint import acb, fmpq

        flint.ctx.prec = prec
        s = self.s
        A, b = self.tableau_generic(x, acb(0), lambda v: acb(fmpq(v.numerator, v.denominator)))
        memo = {}

        def u(t):
            if t in memo:
                return memo[t]
            if not t:
                val = [acb(1)] * s
            else:
                val = None
                for c in t:
                    uc = u(c)
                    w = [sum((A[i][j] * uc[j] for j in range(i) if A[i][j] != 0), acb(0)) for i in range(s)]
                    val = w if val is None else [val[i] * w[i] for i in range(s)]
            memo[t] = val
            return val

        out = []
        for t, ig in zip(self.trees, self.inv_gamma):
            ut = u(t)
            phi = sum((b[i] * ut[i] for i in range(s) if b[i] != 0), acb(0))
            out.append(phi - acb(fmpq(ig.numerator, ig.denominator)))
        return out

    def jacobian_acb(self, x, prec):
        """x: list of acb. Returns (r list of acb, J acb_mat m x n) at `prec` bits (forward mode)."""
        import flint
        from flint import acb, acb_mat, fmpq

        flint.ctx.prec = prec
        s, K = self.s, self.K
        Al, bl = self.tableau_generic(x, acb(0), lambda v: acb(fmpq(v.numerator, v.denominator)))
        A = acb_mat(s, s)
        for i in range(s):
            for j in range(i):
                A[i, j] = Al[i][j]
        memo = {}

        def u(t):
            if t in memo:
                return memo[t]
            if not t:
                val = (acb_mat(s, 1, [1] * s), acb_mat(s, K))
            else:
                ws, dws = [], []
                for c in t:
                    uc, duc = u(c)
                    dA = acb_mat(s, K)
                    for col, (i, j) in enumerate(self.avars):
                        dA[i, col] = uc[j, 0]
                    ws.append(A * uc)
                    dws.append(dA + A * duc)
                mm = len(ws)
                uval = []
                for i in range(s):
                    pr = ws[0][i, 0]
                    for k in range(1, mm):
                        pr = pr * ws[k][i, 0]
                    uval.append(pr)
                du = acb_mat(s, K)
                for k in range(mm):
                    D = acb_mat(s, s)
                    for i in range(s):
                        pr = acb(1)
                        for l in range(mm):
                            if l != k:
                                pr = pr * ws[l][i, 0]
                        D[i, i] = pr
                    du += D * dws[k]
                val = (acb_mat(s, 1, uval), du)
            memo[t] = val
            return val

        bT = acb_mat(1, s, bl)
        rows, r = [], []
        for t, ig in zip(self.trees, self.inv_gamma):
            ut, dut = u(t)
            phi = acb(0)
            for i in range(s):
                phi += bl[i] * ut[i, 0]
            r.append(phi - acb(fmpq(ig.numerator, ig.denominator)))
            ga = bT * dut
            rows.append([ga[0, col] for col in range(K)] + [ut[i, 0] for i in self.bvars])
        J = acb_mat(len(rows), self.n, [v for row in rows for v in row])
        return r, J

    # ---- exact verification modulo f ---------------------------------------------------------------------------
    def verify_exact(self, f, g, fprime=None, progress=None, tree_range=None):
        """f: monic fmpq_poly. g: dict free-name -> fmpq_poly (numerators: x_j = g_j(t) / f'(t) at roots t of f).
        Checks every tree condition exactly in Q[T]/(f): b.u(t) f'(T)^|t| - f'(T)^|t| / gamma(t) = 0 mod f.
        tree_range = (start, end) restricts the check to trees start..end-1 of self.trees (chunked runs).
        Returns (all_zero, list of (order, tree index, residual degree or -1))."""
        from flint import fmpq, fmpq_poly

        if fprime is None:
            fprime = f.derivative()
        s = self.s

        def red(h):
            return h % f

        def const(v):
            return fmpq_poly([fmpq(v.numerator, v.denominator)])

        zero = fmpq_poly([])
        A, b = self.tableau_generic([g[nm] for nm in self.free], zero, lambda v: red(const(v) * fprime))
        fp_pow = [fmpq_poly([1])]
        for _ in range(self.p):
            fp_pow.append(red(fp_pow[-1] * fprime))
        memo = {}

        def N(t):
            if t in memo:
                return memo[t]
            if not t:
                val = [fmpq_poly([1])] * s
            else:
                val = None
                for c in t:
                    uc = N(c)
                    w = []
                    for i in range(s):
                        acc = zero
                        for j in range(i):
                            if not A[i][j].is_zero() and not uc[j].is_zero():
                                acc = acc + A[i][j] * uc[j]
                        w.append(red(acc))
                    val = w if val is None else [red(val[i] * w[i]) for i in range(s)]
            memo[t] = val
            return val

        bad = []
        start, end = tree_range if tree_range is not None else (0, self.m)
        for k in range(start, end):
            t, ig, n = self.trees[k], self.inv_gamma[k], self.tree_order[k]
            ut = N(t)
            acc = zero
            for i in range(s):
                if not b[i].is_zero() and not ut[i].is_zero():
                    acc = acc + b[i] * ut[i]
            res = red(acc - const(ig) * fp_pow[n])
            if not res.is_zero():
                bad.append((n, k, res.degree()))
            if progress and (k + 1 - start) % progress == 0:
                print(f"  verified {k + 1 - start}/{end - start} conditions, failures so far {len(bad)}", flush=True)
        return len(bad) == 0, bad


# ---- rational reconstruction -------------------------------------------------------------------------------------
def reconstruct_rational(value, err, qmax):
    """value: Fraction (an exact decimal read of a floating number); err: absolute error bound (Fraction).
    Returns the unique p/q with q <= qmax and |value - p/q| <= err if one exists (continued fractions), else None.
    Uniqueness holds when 2 * err * qmax^2 < 1 (the caller chooses qmax so)."""
    x = value
    h0, h1 = 1, 0  # numerators of convergents
    k0, k1 = 0, 1  # denominators
    a = x.numerator // x.denominator
    rem = x - a
    h0, h1 = a * h0 + h1, h0
    k0, k1 = a * k0 + k1, k0
    while True:
        if k0 > qmax:
            return None
        cand = Fraction(h0, k0)
        if abs(value - cand) <= err:
            return cand
        if rem == 0:
            return None
        x = 1 / rem
        a = x.numerator // x.denominator
        rem = x - a
        h0, h1 = a * h0 + h1, h0
        k0, k1 = a * k0 + k1, k0


def acb_to_fraction_and_err(z, dps):
    """Real part of an acb ball as (Fraction midpoint, Fraction radius, Fraction bound on |Im z|). The midpoint is
    read to dps + 10 decimal digits. A coefficient whose imaginary part exceeds the error floor is not real: the
    caller treats that as a reconstruction failure (the point set is not closed under conjugation)."""
    re = z.real
    mid = Fraction(re.mid().str(dps + 10, radius=False))
    rad = Fraction(re.rad().str(5, radius=False)) if re.rad() != 0 else Fraction(0)
    im = abs(z.imag)
    imb = Fraction(im.str(5, radius=False)) if im != 0 else Fraction(0)
    return mid, rad + Fraction(1, 10 ** (dps + 8)), imb


def digits(fr):
    """Decimal digits of the larger of numerator and denominator."""
    return max(len(str(abs(fr.numerator))), len(str(fr.denominator)))


# ---- JSON helpers ---------------------------------------------------------------------------------------------------
def acb_str(z, dps):
    return [z.real.mid().str(dps, radius=False), z.imag.mid().str(dps, radius=False)]


def str_acb(pair, prec):
    import flint
    from flint import acb, arb

    flint.ctx.prec = prec
    return acb(arb(pair[0]), arb(pair[1]))


def load_system(path):
    d = json.loads(Path(path).read_text())
    return RKSystem.from_json(d["system"] if "system" in d else d)


def frac_str(fr):
    return f"{fr.numerator}/{fr.denominator}" if fr.denominator != 1 else str(fr.numerator)


def poly_to_json(q):
    return [frac_str(Fraction(int(c.p), int(c.q))) for c in q.coeffs()]


def poly_from_json(lst):
    from flint import fmpq, fmpq_poly

    return fmpq_poly([fmpq(Fraction(v).numerator, Fraction(v).denominator) for v in lst])
