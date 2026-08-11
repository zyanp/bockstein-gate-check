"""Transversal diagonal-gate feasibility check for CSS codes.

Minimal prototype of the decision principle behind arXiv:2602.14499
(J. Haruna, "Homological origin of transversal implementability of
logical diagonal gates in quantum CSS codes").

Setting
-------
A CSS code is given by X-check matrix H_X and Z-check matrix H_Z over
GF(2) with H_X H_Z^T = 0.  A level-m transversal diagonal gate is

    U(theta) = tensor_j diag(1, w^{theta_j}),   w = exp(2 pi i / 2^m),

with a possibly non-uniform angle vector theta in (Z_{2^m})^n.

Decision principle (direct form)
--------------------------------
Codewords are |xbar> = sum_{s in rowspan(H_X)} |l + s> with l a logical-class
representative in ker(H_Z).  Since U is diagonal, U preserves the code space
and acts as a diagonal logical gate iff, for every logical class, the phase

    phi(y) = sum_j theta_j * y_j  (mod 2^m),   y = integer lift of the GF(2) vector,

is constant on the class.  Using the integer lift (l xor s)_j = l_j + s_j - 2 l_j s_j,
constancy on the class of l is equivalent to the linear congruences

    sum_j theta_j * s_j * (1 - 2 l_j) == 0  (mod 2^m)   for all s in rowspan(H_X).

The map s -> lift(l xor s) is NOT linear over GF(2), so we enumerate all
2^{r_X} elements s of rowspan(H_X) (fine for the small codes treated here)
rather than only its generators.  The class representatives are the 2^k
elements of the span of the logical-X coset generators.

The induced logical phase on the class of l is phi(l) = sum_j theta_j l_j
(mod 2^m); on solutions it does not depend on the representative.  A code
admits a *genuinely level-m* transversal diagonal logical gate (logical S for
m = 2, logical T for m = 3, up to odd powers) iff some solution theta gives an
odd phi on some nontrivial logical class.

The solution set of the congruence system is computed exactly by integer
diagonalization (Smith-normal-form style row/column reduction) of the
constraint matrix; no brute force over theta is used.

Pure Python, no external dependencies.
"""

from math import gcd

# ----------------------------------------------------------------------
# GF(2) linear algebra (vectors = lists of 0/1)
# ----------------------------------------------------------------------


def gf2_echelon(rows):
    """Return an echelon basis (list of rows with distinct pivots, sorted by pivot)."""
    basis = []  # list of (pivot, row)
    for row in rows:
        v = list(row)
        for p, b in basis:
            if v[p]:
                v = [x ^ y for x, y in zip(v, b)]
        if any(v):
            piv = next(i for i, x in enumerate(v) if x)
            basis.append((piv, v))
            basis.sort(key=lambda t: t[0])
    return basis


def gf2_reduce(v, basis):
    """Reduce v against an echelon basis (as returned by gf2_echelon)."""
    v = list(v)
    for p, b in basis:
        if v[p]:
            v = [x ^ y for x, y in zip(v, b)]
    return v


def gf2_rank(rows):
    return len(gf2_echelon(rows))


def gf2_kernel_basis(rows, n):
    """Basis of {v in GF(2)^n : M v = 0} for M with the given rows."""
    basis = gf2_echelon(rows)
    pivots = [p for p, _ in basis]
    free = [j for j in range(n) if j not in pivots]
    kernel = []
    for f in free:
        v = [0] * n
        v[f] = 1
        for p, b in sorted(basis, key=lambda t: -t[0]):
            v[p] = sum(b[j] * v[j] for j in range(p + 1, n)) % 2
        kernel.append(v)
    return kernel


def gf2_span(basis_rows, n):
    """All 2^len(basis_rows) elements of the span (includes the zero vector)."""
    out = [[0] * n]
    for b in basis_rows:
        out = out + [[x ^ y for x, y in zip(v, b)] for v in out]
    return out


def logical_x_coset_generators(HX, HZ, n):
    """Generators of ker(H_Z) / rowspan(H_X): representatives of logical-X classes."""
    stab = gf2_echelon(HX)
    gens = []
    gen_basis = []
    for v in gf2_kernel_basis(HZ, n):
        w = gf2_reduce(v, stab)
        w = gf2_reduce(w, gf2_echelon(gen_basis)) if gen_basis else w
        if any(w):
            gens.append(w)
            gen_basis.append(w)
    return gens


# ----------------------------------------------------------------------
# Integer diagonalization (SNF-style) and homogeneous solving mod 2^m
# ----------------------------------------------------------------------


def diagonalize_int(A):
    """Bring integer matrix A to diagonal form D = U A V by unimodular row/column
    operations.  Returns (D, V); U is not tracked (row ops do not change the
    solution set of A theta = 0)."""
    A = [list(row) for row in A]
    r = len(A)
    n = len(A[0]) if r else 0
    V = [[int(i == j) for j in range(n)] for i in range(n)]
    t = 0
    while t < min(r, n):
        # pivot: nonzero entry of minimal |value| in the trailing submatrix
        piv = None
        for i in range(t, r):
            for j in range(t, n):
                a = A[i][j]
                if a and (piv is None or abs(a) < abs(A[piv[0]][piv[1]])):
                    piv = (i, j)
        if piv is None:
            break
        i0, j0 = piv
        A[t], A[i0] = A[i0], A[t]
        if j0 != t:
            for row in A:
                row[t], row[j0] = row[j0], row[t]
            for row in V:
                row[t], row[j0] = row[j0], row[t]
        if A[t][t] < 0:
            for j in range(t, n):
                A[t][j] = -A[t][j]
        p = A[t][t]
        clean = True
        for i in range(t + 1, r):
            if A[i][t]:
                q = A[i][t] // p
                if q:
                    for j in range(t, n):
                        A[i][j] -= q * A[t][j]
                if A[i][t]:
                    clean = False
        for j in range(t + 1, n):
            if A[t][j]:
                q = A[t][j] // p
                if q:
                    for i in range(r):
                        A[i][j] -= q * A[i][t]
                    for i in range(n):
                        V[i][j] -= q * V[i][t]
                if A[t][j]:
                    clean = False
        if clean:
            t += 1
    return A, V


def solve_homogeneous_mod(A, n, q):
    """Generators (over Z_q) of {theta in (Z_q)^n : A theta == 0 mod q}.

    Uses D = U A V: with theta = V theta', the system becomes d_j theta'_j == 0
    (mod q), i.e. theta'_j is a multiple of q / gcd(d_j, q)."""
    if not A:
        return [[int(i == j) for i in range(n)] for j in range(n)]
    D, V = diagonalize_int(A)
    r = len(A)
    gens = []
    for j in range(n):
        d = abs(D[j][j]) if j < r else 0
        c = q // gcd(d, q)  # theta'_j must be a multiple of c
        if c % q == 0:
            continue  # theta'_j == 0 mod q: no generator
        g = [(V[i][j] * c) % q for i in range(n)]
        if any(g):
            gens.append(g)
    return gens


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------


def constraint_rows(HX, HZ, n, m):
    """Constraint matrix rows (over Z, entries in {-1,0,1}) and class reps."""
    stab_elems = gf2_span([b for _, b in gf2_echelon(HX)], n)
    class_gens = logical_x_coset_generators(HX, HZ, n)
    class_reps = gf2_span(class_gens, n)  # 2^k reps, first one is 0
    rows = []
    for l in class_reps:
        for s in stab_elems:
            if not any(s):
                continue
            rows.append([s[j] * (1 - 2 * l[j]) for j in range(n)])
    return rows, class_reps, stab_elems


def analyze(HX, HZ, m):
    """Decide feasibility of a genuinely level-m transversal diagonal logical gate.

    Returns a dict with:
      'gens'        : generators of the solution module of theta (mod 2^m)
      'classes'     : list of (class_rep, phase_subgroup_generator d) for
                      nontrivial logical classes; achievable logical phases on
                      that class form the subgroup d * Z_{2^m}
      'feasible'    : True iff some class admits an odd logical phase (d == 1)
      'example'     : an example theta realizing an odd phase (or None)
      'example_class': the class rep on which the example acts with odd phase
    """
    n = len(HX[0])
    q = 2 ** m
    rows, class_reps, stab_elems = constraint_rows(HX, HZ, n, m)
    gens = solve_homogeneous_mod(rows, n, q)

    # sanity: every generator must satisfy all constraints
    for g in gens:
        for row in rows:
            assert sum(c * x for c, x in zip(row, g)) % q == 0, "solver bug"

    classes = []
    example = None
    example_class = None
    for l in class_reps:
        if not any(l):
            continue
        vals = [sum(l[j] * g[j] for j in range(n)) % q for g in gens]
        d = q
        for v in vals:
            d = gcd(d, v)
        classes.append((l, d))
        if d % 2 == 1 and example is None:
            v = next(v for v in (sum(l[j] * g[j] for j in range(n)) % q
                                 for g in gens) if v % 2 == 1)
            g = next(g for g in gens
                     if sum(l[j] * g[j] for j in range(n)) % q % 2 == 1)
            inv = pow(v, -1, q)
            example = [(x * inv) % q for x in g]
            example_class = l
    feasible = any(d == 1 for _, d in classes)
    return {
        "gens": gens,
        "classes": classes,
        "feasible": feasible,
        "example": example,
        "example_class": example_class,
        "n": n,
        "m": m,
    }


def verify_theta(HX, HZ, m, theta):
    """Independent brute-force verification of a candidate theta.

    Enumerates every codeword coset and checks phase constancy directly on the
    integer lifts (no linear algebra).  Returns {tuple(class_rep): logical phase}
    or raises AssertionError if theta does not preserve the code space."""
    n = len(HX[0])
    q = 2 ** m
    stab_elems = gf2_span([b for _, b in gf2_echelon(HX)], n)
    class_gens = logical_x_coset_generators(HX, HZ, n)
    phases = {}
    for l in gf2_span(class_gens, n):
        ph = {sum(theta[j] * (l[j] ^ s[j]) for j in range(n)) % q
              for s in stab_elems}
        assert len(ph) == 1, "theta does not preserve the code space"
        phases[tuple(l)] = ph.pop()
    assert phases[tuple([0] * n)] == 0
    return phases
