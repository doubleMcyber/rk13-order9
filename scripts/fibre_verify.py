"""Exact verification of a rational univariate representation and the certificate (step 4 of the pipeline).

  .venv/bin/python scripts/fibre_verify.py --rur results/fibre_<tag>_rur.json --tag <tag>
  chunked (runs of bounded length): --tree-range START END writes results/fibre_<tag>_verify_part_START_END.json
  for trees START..END-1 only; then --merge reads every part, checks that the parts cover all trees, and writes
  the certificate.

Reads f (monic, in Q[T]) and the numerators g_j from scripts/fibre_rur.py and proves, in python-flint fmpq_poly
arithmetic modulo f, that the tableau x_j = g_j(T) / f'(T) satisfies every tree condition of the system: for a
tree with n nodes the condition b . u(t) = 1/gamma(t) is multiplied by f'(T)^n, which makes it a polynomial
identity in Q[T] that is checked to reduce to zero modulo f. Fixed coordinates enter as r f'(T). Since f is
squarefree, f'(t) is nonzero at every root t, so the identity modulo f gives the condition exactly at every
root. Real roots are isolated by Sturm sequences: for each real root an interval [a, b] with rational endpoints,
f(a) f(b) < 0 and exactly one root of f in (a, b). The certificate consists of f, the g_j and one isolating
interval; a referee reruns this script (or any computer algebra system) on results/fibre_<tag>_rur.json.
"""
import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fibre_common as fc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sturm_sequence(f):
    seq = [f, f.derivative()]
    while seq[-1].degree() > 0:
        r = seq[-2] % seq[-1]
        if r.is_zero():
            break
        seq.append(-r)
    return seq


def sign_changes(seq, a):
    signs = []
    for q in seq:
        v = q(a)
        if v != 0:
            signs.append(1 if v > 0 else -1)
    return sum(1 for x, y in zip(signs, signs[1:]) if x != y)


def sturm_count(seq, a, b):
    """Number of distinct real roots in (a, b] for a squarefree polynomial."""
    return sign_changes(seq, a) - sign_changes(seq, b)


def isolate_real_roots(f, dps=30):
    """Isolating rational intervals for the real roots of a squarefree f: from the certified complex root balls
    of python-flint, widened to rationals, then confirmed by a sign change and a Sturm count of one."""
    from flint import fmpq

    seq = sturm_sequence(f)
    total = sturm_count(seq, fmpq(-10 ** 9), fmpq(10 ** 9))  # roots are far below 1e9 in size here
    roots = f.complex_roots()
    out = []
    for z, mult in roots:
        if not z.imag.contains(0):
            continue
        re = z.real
        lo = Fraction(re.lower().str(dps, radius=False))
        hi = Fraction(re.upper().str(dps, radius=False))
        width = hi - lo
        a, b = lo - width, hi + width
        fa, fb = f(fmpq(a.numerator, a.denominator)), f(fmpq(b.numerator, b.denominator))
        cnt = sturm_count(seq, fmpq(a.numerator, a.denominator), fmpq(b.numerator, b.denominator))
        out.append({"a": fc.frac_str(a), "b": fc.frac_str(b), "sign_change": (fa < 0) != (fb < 0) and fa != 0 and fb != 0,
                    "sturm_count": cnt, "midpoint": str(re.mid().str(dps, radius=False)), "multiplicity": mult})
    return out, total


def tableau_at_root(system, f, g, interval, dps=40):
    """Evaluate x_j = g_j(t) / f'(t) in arb on the isolating interval (as a ball) for display."""
    import flint
    from flint import arb, arb_poly, fmpq

    flint.ctx.prec = fc.prec_bits(dps)
    a, b = Fraction(interval["a"]), Fraction(interval["b"])
    mid = (a + b) / 2
    rad = (b - a) / 2
    t = arb(arb(fmpq(mid.numerator, mid.denominator)).mid(), arb(fmpq(rad.numerator, rad.denominator)).mid())
    # refine t by interval Newton a few times (f real, simple root)
    fp = arb_poly([arb(fmpq(c.p, c.q)) for c in f.coeffs()])
    fd = fp.derivative()
    for _ in range(6):
        m = t.mid()
        tn = m - fp(m) / fd(t)
        if t.contains(tn) and tn.rad() < t.rad():
            t = tn
    fdt = fd(t)
    vals = {}
    for nm in system.free:
        gj = arb_poly([arb(fmpq(c.p, c.q)) for c in g[nm].coeffs()])
        vals[nm] = gj(t) / fdt
    for nm, v in system.fixed.items():
        vals[nm] = arb(fmpq(v.numerator, v.denominator))
    return t, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rur", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--tree-range", type=int, nargs=2, default=None, help="verify trees START..END-1 only and write a part file")
    ap.add_argument("--merge", action="store_true", help="combine the part files instead of verifying")
    args = ap.parse_args()
    d = json.loads(Path(args.rur).read_text())
    if d.get("f") is None:
        raise SystemExit("the RUR file holds no reconstructed f (reconstruction failed)")
    system = fc.RKSystem.from_json(d["system"])
    f = fc.poly_from_json(d["f"])
    g = {nm: fc.poly_from_json(q) for nm, q in d["g"].items()}
    assert set(g) == set(system.free), "numerators do not match the free coordinates"
    D = f.degree()
    print(f"system s={system.s} p={system.p}, {system.m} conditions, {system.n} free coordinates, D = {D}")
    t0 = time.time()
    fd = f.derivative()
    squarefree = f.gcd(fd).degree() == 0
    monic = f.leading_coefficient() == 1
    print(f"f monic {monic}, squarefree {squarefree}")
    t1 = time.time()
    if args.tree_range is not None:
        a, b = args.tree_range
        ok, bad = system.verify_exact(f, g, fprime=fd, progress=args.progress or None, tree_range=(a, b))
        t_verify = time.time() - t1
        part = {"tag": args.tag, "rur_file": args.rur, "tree_range": [a, b], "all_zero": ok, "failures": bad,
                "seconds": round(t_verify, 2), "f_digest": hash(tuple(d["f"]))}
        out = ROOT / f"results/fibre_{args.tag}_verify_part_{a}_{b}.json"
        out.write_text(json.dumps(part, indent=1) + "\n")
        print(f"trees {a}..{b - 1}: {'ALL ZERO' if ok else f'{len(bad)} FAIL'} ({t_verify:.1f} s) -> {out}")
        return
    if args.merge:
        parts = sorted(ROOT.glob(f"results/fibre_{args.tag}_verify_part_*.json"))
        covered, bad, t_verify = [], [], 0.0
        for pf in parts:
            pd = json.loads(pf.read_text())
            assert pd["rur_file"] == args.rur, f"{pf} was made from {pd['rur_file']}"
            covered.append(tuple(pd["tree_range"]))
            bad += [tuple(x) for x in pd["failures"]]
            t_verify += pd["seconds"]
        covered.sort()
        pos = 0
        for a, b in covered:
            assert a <= pos, f"gap before tree {a}: parts {covered}"
            pos = max(pos, b)
        assert pos >= system.m, f"parts cover only trees 0..{pos - 1} of {system.m}"
        ok = len(bad) == 0
        print(f"merged {len(parts)} parts covering {covered}: {'ALL ZERO' if ok else f'{len(bad)} FAIL'} ({t_verify:.1f} s in all)")
    else:
        ok, bad = system.verify_exact(f, g, fprime=fd, progress=args.progress or None)
        t_verify = time.time() - t1
        print(f"exact verification of all {system.m} conditions modulo f: {'ALL ZERO' if ok else f'{len(bad)} FAIL'} ({t_verify:.1f} s)")
    # fixed coordinates: by construction r f', nothing to check; chart values recorded below
    t2 = time.time()
    intervals, n_real = isolate_real_roots(f)
    t_iso = time.time() - t2
    good = [iv for iv in intervals if iv["sign_change"] and iv["sturm_count"] == 1]
    print(f"real roots: {n_real} by Sturm on (-1e9, 1e9]; {len(good)} isolating intervals certified ({t_iso:.1f} s)")
    cert = {"tag": args.tag, "rur_file": args.rur, "system": system.to_json(), "D": D, "monic": monic, "squarefree": squarefree,
            "all_conditions_zero_mod_f": ok, "n_conditions": system.m, "failures": bad[:20], "n_real_roots": n_real,
            "verified_in_parts": bool(args.merge),
            "real_root_intervals": intervals, "seconds_verify": round(t_verify, 2), "seconds_isolate": round(t_iso, 2),
            "f": d["f"], "weights": d.get("weights"), "certificate": bool(ok and squarefree and monic and good)}
    if good:
        t, vals = tableau_at_root(system, f, g, good[0])
        cert["first_real_root_ball"] = str(t)
        cert["tableau_at_first_real_root"] = {nm: vals[nm].str(30, radius=False) for nm in system.names}
        cert["nodes_at_first_real_root"] = [sum((vals[f"a{i + 1},{j + 1}"] for j in range(i)), vals["b1"] * 0).str(20, radius=False)
                                            for i in range(system.s)]
    out_json = ROOT / f"results/fibre_{args.tag}_certificate.json"
    out_json.write_text(json.dumps(cert, indent=1) + "\n")
    # human-readable certificate
    lines = [f"# Certificate {args.tag}: an exact {system.s}-stage explicit Runge-Kutta method of order {system.p}", "",
             f"Generated by scripts/fibre_verify.py from {args.rur}. Status: "
             f"{'CERTIFIED' if cert['certificate'] else 'NOT CERTIFIED'}.", "",
             f"- Unknowns: {len(system.names)} coordinates (a_ij, j < i, and b_i; nodes c_i = sum_j a_ij). Fixed at rationals: "
             f"{', '.join(f'{k} = {v}' for k, v in d['system']['fixed'].items())}.",
             f"- f(T) is monic of degree D = {D}, squarefree = {squarefree}; every free coordinate is x_j = g_j(T) / f'(T).",
             f"- All {system.m} tree conditions with at most {system.p} nodes, each multiplied by f'(T)^(nodes), reduce to 0 "
             f"modulo f in exact rational arithmetic: {ok}.",
             f"- f has {n_real} real roots; certified isolating intervals (sign change and Sturm count 1): {len(good)}.", ""]
    if good:
        lines += [f"First real root: t in [{good[0]['a']}, {good[0]['b']}] (midpoint about {good[0]['midpoint'][:25]}).", "",
                  "Tableau at that root (30 digits, from the interval; the exact values are g_j(t) / f'(t)):", ""]
        lines += [f"    {nm} = {cert['tableau_at_first_real_root'][nm]}" for nm in system.names]
        lines += ["", "Nodes: " + ", ".join(cert["nodes_at_first_real_root"])]
    lines += ["", "f(T) coefficients, low degree first:", ""] + [f"    {c}" for c in d["f"]]
    lines += ["", f"The numerators g_j are in {args.rur} (key 'g'). To re-verify: rerun this script, or in any computer algebra "
              "system build A and b from g_j / f' and check b . u(t) - 1/gamma(t) = 0 in Q[T]/(f) for every rooted tree t "
              f"with at most {system.p} nodes (definitions in scripts/order_conditions.py)."]
    out_md = ROOT / f"proofs/fibre_{args.tag}_certificate.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"certificate {'OK' if cert['certificate'] else 'FAILED'} -> {out_json}, {out_md}; total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
