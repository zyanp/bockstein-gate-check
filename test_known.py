"""Known-fact tests for the transversal diagonal-gate feasibility checker.

Run:  python test_known.py
Prints a result table and PASS/FAIL lines (ASCII only), exits nonzero on FAIL.

Expected (known) facts:
  Steane  [[7,1,3]] : transversal S  YES   (arXiv:2602.14499, Cor V.1)
  Steane  [[7,1,3]] : transversal T  NO    (arXiv:2602.14499, Cor V.1; holds
                                            even for non-uniform angles)
  RM      [[15,1,3]]: transversal T  YES   (classic result, T^x15 = logical T^dag)
  Surface [[9,1,3]] : transversal T  NO    (known)
  Surface [[9,1,3]] : transversal S  NO    (claim to be verified; grant
                                            application Fig. 4)
"""

import sys
import time

from bockstein_check import analyze, verify_theta
from codes import ALL_CODES, check_code, steane, reed_muller_15

GATE_NAME = {1: "Z", 2: "S", 3: "T"}

failures = []


def report(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print("%s  %s%s" % (status, label, ("  [" + detail + "]") if detail else ""))
    if not ok:
        failures.append(label)


def main():
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Code self-checks
    # ------------------------------------------------------------------
    print("== Code self-checks ==")
    codes = []
    for name, ctor, (n, k, d) in ALL_CODES:
        HX, HZ = ctor()
        try:
            check_code(name, HX, HZ, n, k, d)
            report("%s is a valid [[%d,%d,%d]] CSS code" % (name, n, k, d), True)
        except AssertionError as e:
            report(name + " self-check", False, str(e))
        codes.append((name, HX, HZ))

    # ------------------------------------------------------------------
    # Feasibility analysis: each code at levels m = 1, 2, 3
    # ------------------------------------------------------------------
    print()
    print("== Feasibility (genuinely level-m transversal diagonal logical gate) ==")
    results = {}
    for name, HX, HZ in codes:
        for m in (1, 2, 3):
            res = analyze(HX, HZ, m)
            results[(name, m)] = res
            # cross-check any feasible example by brute-force phase enumeration
            if res["feasible"]:
                phases = verify_theta(HX, HZ, m, res["example"])
                ph = phases[tuple(res["example_class"])]
                assert ph % 2 == 1, "example does not realize an odd phase"

    header = "%-26s %-6s %-9s %-16s %s" % ("code", "gate", "feasible",
                                           "phase subgroup", "example theta")
    print(header)
    print("-" * len(header))
    for name, HX, HZ in codes:
        for m in (1, 2, 3):
            res = results[(name, m)]
            q = 2 ** m
            d = min((dd for _, dd in res["classes"]), default=q)
            sub = "{0}" if d == q else "%d*Z_%d" % (d, q)
            ex = str(res["example"]) if res["example"] else "-"
            print("%-26s %-6s %-9s %-16s %s" % (
                name, GATE_NAME[m] + " (m=%d)" % m,
                "YES" if res["feasible"] else "NO", sub, ex))

    # ------------------------------------------------------------------
    # Known-fact assertions
    # ------------------------------------------------------------------
    print()
    print("== Known-fact tests ==")
    st, rm, sf = codes[0][0], codes[1][0], codes[2][0]

    for name in (st, rm, sf):
        report("%s: transversal logical Z (m=1) feasible (sanity)" % name,
               results[(name, 1)]["feasible"])

    report("Steane: transversal S feasible (arXiv:2602.14499 Cor V.1)",
           results[(st, 2)]["feasible"])
    report("Steane: transversal T infeasible even non-uniform (Cor V.1)",
           not results[(st, 3)]["feasible"])
    report("RM15: transversal T feasible (known)",
           results[(rm, 3)]["feasible"])
    report("Surface: transversal T infeasible (known)",
           not results[(sf, 3)]["feasible"])
    report("Surface: transversal S infeasible (application Fig.4 claim)",
           not results[(sf, 2)]["feasible"])

    # ------------------------------------------------------------------
    # Direct reproduction of the canonical gate implementations
    # ------------------------------------------------------------------
    print()
    print("== Canonical implementations (independent brute-force check) ==")

    # Steane: U(3 * 1_7) at m=2, i.e. (S^dag)^x7, is a logical S (phase i on |1bar>)
    HX, HZ = steane()
    try:
        phases = verify_theta(HX, HZ, 2, [3] * 7)
        ph = [p for c, p in phases.items() if any(c)][0]
        report("Steane: theta = 3*1_7 (S^dag each) gives odd logical phase %d mod 4" % ph,
               ph % 2 == 1)
    except AssertionError as e:
        report("Steane: theta = 3*1_7 preserves code space", False, str(e))

    # RM15: U(1_15) at m=3, i.e. T^x15, is a logical T^dag (odd phase on |1bar>)
    HX, HZ = reed_muller_15()
    try:
        phases = verify_theta(HX, HZ, 3, [1] * 15)
        ph = [p for c, p in phases.items() if any(c)][0]
        report("RM15: theta = 1_15 (T each) gives odd logical phase %d mod 8" % ph,
               ph % 2 == 1)
    except AssertionError as e:
        report("RM15: theta = 1_15 preserves code space", False, str(e))

    dt = time.perf_counter() - t0
    print()
    print("Total time: %.3f s" % dt)
    if failures:
        print("RESULT: FAIL (%d failed)" % len(failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
