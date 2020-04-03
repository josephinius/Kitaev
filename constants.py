import math
import numpy as np
from scipy import linalg

EPS = 1.E-32


def get_spin_operators(spin):
    """Returns tuple of 3 spin operators and a unit matrix for given value of spin."""

    if type(spin) == str and len(spin) > 1:
        spin_float = float(spin[0]) / float(spin[2])
        s = float(spin_float)
    else:
        s = float(spin)

    d = int(2 * s + 1)
    eye = np.eye(d, dtype=complex)

    sx = np.zeros([d, d], dtype=complex)
    Sy = np.zeros([d, d], dtype=complex)
    Sz = np.zeros([d, d], dtype=complex)

    for a in range(d):
        if a != 0:
            sx[a, a - 1] = np.sqrt((s + 1) * (2 * a) - (a + 1) * a) / 2
            Sy[a, a - 1] = 1j * np.sqrt((s + 1) * (2 * a) - (a + 1) * a) / 2
        if a != d - 1:
            sx[a, a + 1] = np.sqrt((s + 1) * (2 * a + 2) - (a + 2) * (a + 1)) / 2
            Sy[a, a + 1] = -1j * np.sqrt((s + 1) * (2 * a + 2) - (a + 2) * (a + 1)) / 2
        Sz[a, a] = s - a

    if spin == '1/2':
        sx *= 2
        Sy *= 2
        Sz *= 2

    return sx, Sy, Sz, eye


"""
def get_spin_operators(spin):
    # Returns tuple of 3 spin operators and a unit matrix for given value of spin.
    if spin == "1/2":
        return sx12, sy12, sz12, np.eye(2)
    elif spin == "1":
        return SX1, SY1, SZ1, np.eye(3)
    elif spin == '3/2':
        return sx32, sy32, sz32, np.eye(4)
    elif spin == '2':
        return SX2, SY2, SZ2, np.eye(5)
    elif spin == '5/2':
        return sx52, sy52, sz52, np.eye(6)
    elif spin == '3':
        return SX3, SY3, SZ3, np.eye(7)
    raise ValueError('Supported spin-values: "1/2", "1", "3/2", "2", "5/2", "3".')
"""

# Spin-1/2 operators

sx12 = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

sy12 = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

sz12 = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)


# Spin=1 operators

SX1 = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=complex) / math.sqrt(2)

SY1 = np.array([
    [0, 1, 0],
    [-1, 0, 1],
    [0, -1, 0]
], dtype=complex) / (math.sqrt(2) * 1j)

SZ1 = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, -1]
], dtype=complex)


# Spin-3/2 operators

sx32 = np.array([
    [0, np.sqrt(3), 0, 0],
    [np.sqrt(3), 0, 2, 0],
    [0, 2, 0, np.sqrt(3)],
    [0, 0, np.sqrt(3), 0]
], dtype=complex) / 2

sy32 = np.array([
    [0, -np.sqrt(3) * 1j, 0, 0],
    [np.sqrt(3) * 1j, 0, -2j, 0],
    [0, 2j, 0, -np.sqrt(3) * 1j],
    [0, 0, np.sqrt(3) * 1j, 0]
], dtype=complex)/2

sz32 = np.array([
    [3, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -3]
], dtype=complex)/2


# Spin-2 operators

SX2 = np.array([
    [0, 2, 0, 0, 0],
    [2, 0, np.sqrt(6), 0, 0],
    [0, np.sqrt(6), 0, np.sqrt(6), 0],
    [0, 0, np.sqrt(6), 0, 2],
    [0, 0, 0, 2, 0]
], dtype=complex) / 2

SY2 = np.array([
    [0, -2j, 0, 0, 0],
    [2j, 0, -np.sqrt(6) * 1j, 0, 0],
    [0, np.sqrt(6) * 1j, 0, -np.sqrt(6) * 1j, 0],
    [0, 0, np.sqrt(6) * 1j, 0, -2j],
    [0, 0, 0, 2j, 0]
], dtype=complex) / 2

SZ2 = np.zeros([5, 5], dtype=complex)
SZ2[0, 0] = 2
SZ2[1, 1] = 1
SZ2[2, 2] = 0
SZ2[3, 3] = -1
SZ2[4, 4] = -2


# Spin-5/2 operators

sx52 = np.zeros([6, 6], dtype=complex)
sx52[0, 1] = sx52[1, 0] = sx52[4, 5] = sx52[5, 4] = np.sqrt(5) / 2
sx52[1, 2] = sx52[2, 1] = sx52[3, 4] = sx52[4, 3] = np.sqrt(2)
sx52[2, 3] = sx52[3, 2] = 1.5

sy52 = np.zeros([6, 6], dtype=complex)
sy52[0, 1] = sy52[4, 5] = -1j * np.sqrt(5) / 2
sy52[1, 0] = sy52[5, 4] = 1j * np.sqrt(5) / 2
sy52[1, 2] = sy52[3, 4] = -1j * np.sqrt(2)
sy52[2, 1] = sy52[4, 3] = 1j * np.sqrt(2)
sy52[2, 3] = -1.5 * 1j
sy52[3, 2] = 1.5 * 1j

sz52 = np.array([5, 3, 1, -1, -3, -5], dtype=complex) / 2
sz52 = np.diag(sz52)


# Spin-3 operators

SX3 = np.zeros([7, 7], dtype=complex)
SY3 = np.zeros([7, 7], dtype=complex)
SZ3 = np.zeros([7, 7], dtype=complex)
for a in range(7):
    if a != 0:
        SX3[a, a - 1] = np.sqrt(4 * (2 * a) - (a + 1) * a) / 2
        SY3[a, a - 1] = 1j * np.sqrt(4 * (2 * a) - (a + 1) * a) / 2
    if a != 6:
        SX3[a, a + 1] = np.sqrt(4 * (2 * a + 2) - (a + 2) * (a + 1)) / 2
        SY3[a, a + 1] = -1j * np.sqrt(4 * (2 * a + 2) - (a + 2) * (a + 1)) / 2
    SZ3[a, a] = 4 - (a + 1)


UX = np.array([
    [0, 0, -1.],
    [0, -1., 0],
    [-1., 0, 0]
], dtype=complex)


UY = np.array([
    [0, 0, 1.],
    [0, -1., 0],
    [1., 0, 0]
], dtype=complex)


UZ = np.array([
    [-1., 0, 0],
    [0, 1., 0],
    [0, 0, -1.]
], dtype=complex)


# UX = linalg.expm(1j * math.pi * SX)
# UX = exponentiation(1j * math.pi, SX)
# UY = linalg.expm(1j * math.pi * SY)
# UY = exponentiation(1j * math.pi, SY)
# UZ = linalg.expm(1j * math.pi * SZ)
# UZ = exponentiation(1j * math.pi, SZ)


# Spin=1 Kitaev model - magnetized (polarized) state:
mag_state_s1_kitaev = (- 1j * (2 + math.sqrt(3)), (1 - 1j) * (math.sqrt(2) + math.sqrt(6)) / 2, 1)


def create_loop_gas_operator(spin):
    """Returns loop gas (LG) operator Q_LG for spin=1/2 or spin=1 Kitaev model."""

    tau_tensor = np.zeros((2, 2, 2), dtype=complex)  # tau_tensor_{i j k}

    if spin == "1/2" or spin == '3/2' or spin == '5/2':
        tau_tensor[0][0][0] = - 1j
    elif spin == "1" or spin == '2' or spin == '3':
        tau_tensor[0][0][0] = 1

    tau_tensor[0][1][1] = tau_tensor[1][0][1] = tau_tensor[1][1][0] = 1

    sx, sy, sz, one = get_spin_operators(spin)
    d = one.shape[0]

    Q_LG = np.zeros((d, d, 2, 2, 2), dtype=complex)  # Q_LG_{s s' i j k}

    u_gamma = None

    if spin == "1/2":
        u_gamma = (sx, sy, sz)
    elif spin == '3/2' or spin == '5/2':
        u_gamma = tuple(map(lambda x: -1j * linalg.expm(1j * math.pi * x), (sx, sy, sz)))
    elif spin == "1" or spin == '2' or spin == '3':
        u_gamma = tuple(map(lambda x: linalg.expm(1j * math.pi * x), (sx, sy, sz)))

    for i in range(2):
        for j in range(2):
            for k in range(2):
                temp = np.eye(d)
                if i == 0:
                    temp = temp @ u_gamma[0]
                if j == 0:
                    temp = temp @ u_gamma[1]
                if k == 0:
                    temp = temp @ u_gamma[2]
                for s in range(d):
                    for sp in range(d):
                        Q_LG[s][sp][i][j][k] = tau_tensor[i][j][k] * temp[s][sp]

    return Q_LG


def exponentiation(alpha, s):
    """
    Returns matrix exponentiation exp(alpha * s), where alpha is (complex) coefficient and s is (Hermitian) matrix.

    Note: This method of matrix exponentiation is not numerically accurate and is used for debugging purposes only.
    """

    w, v = linalg.eigh(s)
    # w, v = linalg.eig(s)
    d = s.shape[0]
    assert np.allclose(s @ v - v @ np.diag(w), np.zeros((d, d)), rtol=1e-12, atol=1e-14)
    assert np.allclose(s - v @ np.diag(w) @ linalg.inv(v), np.zeros((d, d)), rtol=1e-12, atol=1e-14)
    assert np.allclose(v @ linalg.inv(v), np.eye(d), rtol=1e-12, atol=1e-14)
    # w = np.where(np.abs(np.imag(w)) < 1.E-14, np.real(w), w)
    # w = np.where(np.abs(np.real(w)) < 1.E-14, np.imag(w), w)
    # print('w', w)
    result = v @ np.exp(alpha * np.diag(w)) @ linalg.inv(v)
    # return v @ np.exp(np.diag(alpha * w)) @ linalg.inv(v)
    # result = np.where(np.abs(result) < 5.E-14, 0, result)
    # result = np.where(np.abs(np.imag(result)) < 1.E-15, np.real(result), result)
    # result = np.where(np.abs(np.real(result)) < 1.E-15, np.imag(result), result)
    return result
