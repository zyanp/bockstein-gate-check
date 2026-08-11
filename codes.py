"""Definitions of the three CSS codes used in the known-fact tests, with
self-checks of H_X H_Z^T = 0 and the [[n, k, d]] parameters.

Pure Python, no external dependencies.
"""

from bockstein_check import gf2_echelon, gf2_rank, gf2_reduce, gf2_kernel_basis, gf2_span


def steane():
    """Steane code [[7,1,3]]: H_X = H_Z = parity-check matrix of the [7,4]
    Hamming code (column j = binary representation of j+1)."""
    H = [[(j + 1) >> i & 1 for j in range(7)] for i in range(3)]
    return H, [row[:] for row in H]


def reed_muller_15():
    """Quantum Reed-Muller code [[15,1,3]] (punctured RM(1,4) construction).

    X checks: 4 rows v_i with v_i[j] = bit i of (j+1), j = 0..14 (columns are
    the binary representations of 1..15).
    Z checks: the same 4 rows plus the 6 componentwise products v_i * v_j
    (i < j), i.e. the even-weight subcode of punctured RM(2,4)."""
    v = [[(j + 1) >> i & 1 for j in range(15)] for i in range(4)]
    HX = [row[:] for row in v]
    HZ = [row[:] for row in v]
    for i in range(4):
        for j in range(i + 1, 4):
            HZ.append([a & b for a, b in zip(v[i], v[j])])
    return HX, HZ


def rotated_surface_9():
    """Rotated surface code [[9,1,3]] on a 3x3 grid of data qubits:

        0 1 2
        3 4 5
        6 7 8

    X stabilizers: {1,2,4,5}, {3,4,6,7}, {0,1}, {7,8}
    Z stabilizers: {0,1,3,4}, {4,5,7,8}, {2,5}, {3,6}
    """
    def rows(supports):
        return [[int(j in s) for j in range(9)] for s in supports]

    HX = rows([{1, 2, 4, 5}, {3, 4, 6, 7}, {0, 1}, {7, 8}])
    HZ = rows([{0, 1, 3, 4}, {4, 5, 7, 8}, {2, 5}, {3, 6}])
    return HX, HZ


# ----------------------------------------------------------------------
# Self-checks
# ----------------------------------------------------------------------


def _min_logical_weight(H_ker, H_stab, n):
    """Min weight over ker(H_ker) \\ rowspan(H_stab) (a code distance)."""
    stab = gf2_echelon(H_stab)
    kernel = gf2_kernel_basis(H_ker, n)
    best = None
    for v in gf2_span(kernel, n):
        if not any(v):
            continue
        if not any(gf2_reduce(v, stab)):
            continue  # v is a stabilizer, not a logical
        w = sum(v)
        if best is None or w < best:
            best = w
    return best


def check_code(name, HX, HZ, expect_n, expect_k, expect_d):
    """Verify CSS orthogonality and the [[n,k,d]] parameters. Returns (n,k,d)."""
    n = len(HX[0])
    assert len(HZ[0]) == n, name + ": H_X and H_Z have different n"
    for x in HX:
        for z in HZ:
            assert sum(a & b for a, b in zip(x, z)) % 2 == 0, \
                name + ": H_X H_Z^T != 0"
    k = n - gf2_rank(HX) - gf2_rank(HZ)
    dX = _min_logical_weight(HZ, HX, n)  # weight of minimal X-type logical
    dZ = _min_logical_weight(HX, HZ, n)  # weight of minimal Z-type logical
    d = min(dX, dZ)
    assert (n, k, d) == (expect_n, expect_k, expect_d), \
        "%s: got [[%d,%d,%d]], expected [[%d,%d,%d]]" % (
            name, n, k, d, expect_n, expect_k, expect_d)
    return n, k, d


ALL_CODES = [
    ("Steane [[7,1,3]]", steane, (7, 1, 3)),
    ("Reed-Muller [[15,1,3]]", reed_muller_15, (15, 1, 3)),
    ("Rotated surface [[9,1,3]]", rotated_surface_9, (9, 1, 3)),
]


if __name__ == "__main__":
    for name, ctor, (n, k, d) in ALL_CODES:
        HX, HZ = ctor()
        check_code(name, HX, HZ, n, k, d)
        print("OK  %s  self-check passed" % name)
