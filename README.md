# rk13-order9

Exact data and checking scripts for M. Moenius, "An explicit 13-stage Runge-Kutta method of order 9 with exact
algebraic coefficients" (`paper/rk13-order9.pdf`).

The paper gives an explicit 13-stage Runge-Kutta method of order 9 whose coefficients lie in a totally real number
field of degree 16, and verifies all 486 order conditions in exact rational arithmetic. This proves a(9) <= 13 for
the minimal number of stages (OEIS A187102).

To recheck the proof with nothing but a Python 3 interpreter (about a minute):

    python3 scripts/purepy_exact_check.py results/sigma_rur_inner1_tower.json

`README.txt` describes every file and the other checks, and `SHA256SUMS` lists the SHA-256 digest of every file
except the PDF and this README.
