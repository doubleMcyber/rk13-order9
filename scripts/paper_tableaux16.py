"""The certified tableau at all 16 real embeddings of K, to 30 significant digits, for the paper's data deposit (2026-11-01).

  .venv/bin/python scripts/paper_tableaux16.py results/sigma_rur_inner1_tower.json results/fibre_sigma_inner1_certificate.json

Reads the exact univariate form (f, g_j) and the 16 certified isolating intervals of the real roots of f, narrows each interval by exact bisection to width
1e-200, evaluates every coefficient x_j = g_j(theta) / f'(theta) in arb ball arithmetic there (fibre_verify.tableau_at_root, at 120
digits), asserts that every ball is narrow enough for its 30 printed digits, and, as a numerical check
that is not part of the proof, the 486 order-condition residuals there. Writes results/sigma_inner1_tableaux_16_embeddings.json.
"""
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

from flint import fmpq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fibre_common as fc  # noqa: E402
from fibre_verify import tableau_at_root  # noqa: E402

DPS = 120
WIDTH = Fraction(1, 10 ** 200)


def narrow(f, iv):
    """Bisect the certified isolating interval [a, b] in exact rational arithmetic (sign of f) to width below WIDTH."""
    a, b = Fraction(iv["a"]), Fraction(iv["b"])
    fa = f(fmpq(a.numerator, a.denominator))
    assert fa != 0 and fa * f(fmpq(b.numerator, b.denominator)) < 0, "no sign change on the interval"
    while b - a > WIDTH:
        m = (a + b) / 2
        fm = f(fmpq(m.numerator, m.denominator))
        if fm == 0:
            return {"a": str(m), "b": str(m)}
        if (fm < 0) == (fa < 0):
            a, fa = m, fm
        else:
            b = m
    return {"a": str(a), "b": str(b)}


def main():
    t_all = time.time()
    rur_path, cert_path = sys.argv[1], sys.argv[2]
    d = json.loads(Path(rur_path).read_text())
    cert = json.loads(Path(cert_path).read_text())
    assert cert["rur_file"] == rur_path, f"certificate was made from {cert['rur_file']}"
    system = fc.RKSystem.from_json(d["system"])
    f = fc.poly_from_json(d["f"])
    g = {nm: fc.poly_from_json(q) for nm, q in d["g"].items()}
    good = [iv for iv in cert["real_root_intervals"] if iv["sign_change"] and iv["sturm_count"] == 1]
    assert len(good) == 16, f"{len(good)} certified intervals"
    from flint import acb
    out = {"source": rur_path, "certificate": cert_path, "dps": DPS, "digits_shown": 30,
           "note": "Coefficients at each real embedding theta of f (exact values g_j(theta)/f'(theta)); residuals are a numerical check, not the proof.",
           "embeddings": []}
    for k, iv in enumerate(good):
        theta, vals = tableau_at_root(system, f, g, narrow(f, iv), dps=DPS)
        worst = max(float(vals[nm].rad()) / max(float(abs(vals[nm].mid())), 1e-300) for nm in system.free)
        assert worst < 1e-35, f"embedding {k}: relative ball radius {worst:.1e}, so 30 digits are not certified"
        res = system.residual_acb([acb(vals[nm]) for nm in system.free], fc.prec_bits(DPS))
        rmax = max(abs(r).upper() for r in res)
        nodes = [sum((vals[f"a{i + 1},{j + 1}"] for j in range(i)), vals["b1"] * 0) for i in range(system.s)]
        out["embeddings"].append({"index": k, "theta_interval": [iv["a"], iv["b"]], "theta": theta.str(30, radius=False),
                                  "coefficients": {nm: vals[nm].str(30, radius=False) for nm in system.names},
                                  "nodes": [c.str(30, radius=False) for c in nodes],
                                  "max_relative_radius": f"{worst:.1e}",
                                  "max_residual_upper_bound": rmax.str(5, radius=False)})
        print(f"embedding {k}: theta {theta.str(15, radius=False)}, max relative radius {worst:.1e}, max residual <= {rmax.str(5, radius=False)}", flush=True)
    out["seconds"] = round(time.time() - t_all, 1)
    p = fc.ROOT / "results/sigma_inner1_tableaux_16_embeddings.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"done: 16 embeddings, {out['seconds']} s -> {p}")


if __name__ == "__main__":
    main()
