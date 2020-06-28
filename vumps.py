import numpy as np
import cmath
from ncon import ncon
from scipy import linalg
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicgstab
from scipy.linalg import polar

from constants import sx12 as SX
# from constants import sy12 as SY
from constants import sz12 as SZ

"""

References:

[1] Phys. Rev. B 97, 045145 (2018)

"""


class NoConvergenceError(Exception):
    pass


def apply_transfer_matrix(y, a):
    """Returns transfer matrix a * conj(a) applied to y"""

    ya = np.tensordot(y, a, axes=(1, 0))
    result = np.tensordot(ya, np.conj(a), axes=([0, 1], [0, 1]))
    return result


def half_infinity_sum_iterative(h, a, eps_sum=1.E-16):
    """Calculates H_L (or H_R) iteratively, see Eq. (14) in Ref. [1]."""

    # TODO: test against function half_infinity_sum_explicit

    accuracy, prev_accuracy = 1, float('inf')
    h_sum = h
    i = 0
    while accuracy > eps_sum:
        ht = apply_transfer_matrix(h_sum, a)
        h_sum_new = ht + h
        if i > 10 and prev_accuracy + eps_sum < accuracy:
            raise NoConvergenceError(f'Iterative update seems to diverge; number of iterations: {i}')
        prev_accuracy = accuracy
        accuracy = linalg.norm(h_sum - h_sum_new) / linalg.norm(h_sum_new)
        h_sum = h_sum_new
        i += 1

    print(f'half_infinity_sum_iterative after {i} iterations')

    return h_sum


def create_map_y(a, r):
    dim = a.shape[0]

    def inner(y):
        y = y.reshape(dim, dim)
        yt = apply_transfer_matrix(y, a)
        yr = np.tensordot(y, r, axes=([0, 1], [0, 1])) * np.eye(dim)
        return (y - yt + yr).reshape(-1)

    return inner


def half_infinity_sum_explicit(h, a, r, eps_sum=1.E-16):  # r stands for reduced density matrix (L, R)
    map_y = create_map_y(a, r)
    dim = a.shape[0]
    right_hand_side = h - np.tensordot(h, r, axes=([0, 1], [0, 1])) * np.eye(dim)
    right_hand_side = right_hand_side.reshape(-1)
    linear_system = LinearOperator((dim ** 2, dim ** 2), matvec=map_y)
    y, info = bicgstab(linear_system, right_hand_side, x0=right_hand_side, tol=eps_sum)

    if info > 0:
        raise NoConvergenceError(f'Convergence to tolerance not achieved, number of iterations: {info}')
    if info < 0:
        raise ValueError('Illegal input or breakdown')

    return y.reshape(dim, dim)


def create_h_l(a_l, h):
    """Returns h_l constructed according to Eq. (12) from Ref. [1]."""

    h_l = ncon(
        [a_l,
         a_l,
         h,
         np.conj(a_l),
         np.conj(a_l)],
        [[1, 3, 2],
         [2, 4, -2],
         [3, 4, 5, 6],
         [1, 5, 7],
         [7, 6, -1]]
    )

    return h_l


def create_h_r(a_r, h):
    """Returns h_r constructed according to Eq. (12) from Ref. [1]."""

    h_r = ncon(
        [a_r,
         a_r,
         h,
         np.conj(a_r),
         np.conj(a_r)],
        [[2, 3, -2],
         [1, 4, 2],
         [3, 4, 5, 6],
         [7, 5, -1],
         [1, 6, 7]]
    )

    return h_r


def umps_to_lam_gamma_form(a):
    """Returns (\lambda, \Gamma) representation of uMPS |\Psi(a)> given by local tensor a."""

    dim, d, _ = a.shape
    u, s, vt = linalg.svd(a.reshape(dim * d, dim), full_matrices=False)
    lam = np.diag(s) / linalg.norm(s)
    gamma = ncon([vt, u.reshape(dim, d, dim)],
                 [[-1, 1], [1, -2, -3]])
    norm = ncon([lam @ lam, gamma, lam @ lam, np.conj(gamma)],
                [[1, 4], [1, 3, 2], [2, 5], [4, 3, 5]])
    gamma = gamma / np.sqrt(norm)
    return lam, gamma


def create_transfer_map_l(lam, gamma):
    dim = lam.shape[0]

    def inner(l):
        l = l.reshape(dim, dim)
        l_out = ncon([l, np.conj(lam), np.conj(gamma), lam, gamma],
                     [[1, 2], [1, 3], [3, 5, -1], [2, 4], [4, 5, -2]])
        return l_out

    return inner


def create_transfer_map_r(lam, gamma):
    dim = lam.shape[0]

    def inner(r):
        r = r.reshape(dim, dim)
        r_out = ncon([gamma, lam, np.conj(gamma), np.conj(lam), r],
                     [[-2, 1, 2], [2, 3], [-1, 1, 4], [4, 5], [5, 3]])
        return r_out

    return inner


def calculate_L(l):
    """Decompose Hermitian matrix l as l = Y^{\dagger} Y as follows:

    l = u w u^{\dagger}  # eigenvalue decomposition
    y = sqrt(w) u^{dagger}
    y^{\dagger} = u sqrt(w)

    """

    w, u = linalg.eigh(l)
    w_sqrt = np.diag(list(map(cmath.sqrt, w)))
    u_dagger = np.conj(u.T)
    return w_sqrt @ u_dagger


def calculate_R(r):
    """Decompose Hermitian matrix r as r = Y Y^{\dagger} as follows:

    r = u w u^{\dagger}  # eigenvalue decomposition
    y = u sqrt(w)
    y^{\dagger} = sqrt(w) u^{dagger}

    """

    w, u = linalg.eigh(r)
    w_sqrt = np.diag(list(map(cmath.sqrt, w)))
    return u @ w_sqrt


def lam_gamma_to_canonical(lam, gamma):
    """Transform general (\lambda, \Gamma) into canonical form."""

    transfer_map_l = create_transfer_map_l(lam, gamma)
    transfer_map_r = create_transfer_map_r(lam, gamma)

    dim, d, _ = gamma.shape

    l = np.random.rand(dim ** 2)
    _, l = eigs(LinearOperator((dim ** 2, dim ** 2), matvec=transfer_map_l), k=1, which='LM', v0=l)
    l = l.reshape(dim, dim)

    r = np.random.rand(dim ** 2)
    _, r = eigs(LinearOperator((dim ** 2, dim ** 2), matvec=transfer_map_r), k=1, which='LM', v0=r)
    r = r.reshape(dim, dim)

    L = calculate_L(l)
    R = calculate_R(r)

    u, s, vt = linalg.svd(L @ lam @ R, full_matrices=False)
    lam_new = np.diag(s)

    gamma_new = ncon([vt @ linalg.inv(R), gamma, linalg.inv(L) @ u],
                     [[-1, 1], [1, -2, 2], [2, -3]])

    lam_new = lam_new / linalg.norm(lam_new)

    norm = ncon([lam_new @ lam_new, gamma_new, lam_new @ lam_new, np.conj(gamma_new)],
                [[1, 4], [1, 3, 2], [2, 5], [4, 3, 5]])

    gamma_new = gamma_new / np.sqrt(norm)

    return lam_new, gamma_new


def canonical_to_left_and_right_canonical(lam, gamma):
    """Returns left and right canonical form using canonical (\lambda, \Gamma) form."""

    left_canonical = ncon([lam, gamma],
               [[-1, 1], [1, -2, -3]])
    right_canonical = ncon([gamma, lam],
               [[-3, -2, 1], [1, -1]])
    return left_canonical, right_canonical


def evaluate_energy_two_sites(A_L, A_R, Ac, h):  # TODO: clean this function

    e1 = ncon([A_L, Ac, h, np.conj(A_L), np.conj(Ac)],
              [[1, 4, 2], [2, 5, 3], [4, 5, 6, 7], [1, 6, 8], [8, 7, 3]])
    e2 = ncon([Ac, A_R, h, np.conj(Ac), np.conj(A_R)],
              [[1, 4, 2], [3, 5, 2], [4, 5, 6, 7], [1, 6, 8], [3, 7, 8]])

    e = (e1 + e2) / 2
    # print('abs(e-e1) = ', abs(e-e1))

    if abs(e - e1) > 1e-14:
        print('e1 is not close to e2!')
        print('abs(e-e1) = ', abs(e - e1))
        # exit()

    return e


def create_Hac_map(A_L, A_R, L_h, R_h, h):

    dim, d, _ = A_L.shape

    def inner(Ac):
        Ac = Ac.reshape(dim, d, dim)
        term1 = ncon([A_L, Ac, h, np.conj(A_L)],
                     [[5, 2, 1], [1, 3, -3], [2, 3, 4, -2], [5, 4, -1]])
        term2 = ncon([Ac, A_R, h, np.conj(A_R)],
                     [[-1, 2, 1], [5, 3, 1], [2, 3, -2, 4], [5, 4, -3]])
        term3 = ncon([L_h, Ac],
                     [[-1, 1], [1, -2, -3]])
        term4 = ncon([Ac, R_h],
                     [[-1, -2, 1], [-3, 1]])
        final = term1 + term2 + term3 + term4
        return final.reshape(-1)

    return inner


def create_Hc_map(A_L, A_R, L_h, R_h, h):

    dim = A_L.shape[0]

    def inner(C):
        C = C.reshape(dim, dim)
        term1 = ncon([A_L, C, A_R, h, np.conj(A_L), np.conj(A_R)],
                     [[1, 5, 2],
                      [2, 3],
                      [4, 6, 3],
                      [5, 6, 7, 8],
                      [1, 7, -1],
                      [4, 8, -2]])
        term2 = L_h @ C
        term3 = C @ R_h.T
        final = term1 + term2 + term3
        return final.reshape(-1)

    return inner


def min_Ac_C(Ac, C):

    print('In min_Ac_C...')

    dim, d, _ = Ac.shape

    U_Ac, S_Ac, V_dagger_Ac = linalg.svd(Ac.reshape(dim * d, dim), full_matrices=False)
    U_c, S_c, V_dagger_c = linalg.svd(C, full_matrices=False)

    A_L = (U_Ac @ V_dagger_Ac) @ np.conj(U_c @ V_dagger_c).T
    A_L = A_L.reshape(dim, d, dim)

    Ac_r = Ac.transpose([2, 1, 0])
    C_r = C.T

    U_Ac, S_Ac, V_dagger_Ac = linalg.svd(Ac_r.reshape(dim * d, dim), full_matrices=False)

    U_c, S_c, V_dagger_c = linalg.svd(C_r, full_matrices=False)
    A_R = (U_Ac @ V_dagger_Ac) @ np.conj(U_c @ V_dagger_c).T
    A_R = A_R.reshape(dim, d, dim)

    u_l_ac, _ = polar(Ac.reshape(dim * d, dim), side='left')
    u_l_c, _ = polar(C, side='left')

    al_tilda = u_l_ac @ np.conj(u_l_c).T
    al_tilda = al_tilda.reshape(dim, d, dim)

    print('left test:', linalg.norm(al_tilda - A_L))

    u_r_ac, _ = polar(Ac.reshape(dim, dim * d), side='right')
    u_r_c, _ = polar(C, side='right')

    ar_tilda = np.conj(u_r_c).T @ u_r_ac
    ar_tilda = ar_tilda.reshape(dim, d, dim)
    ar_tilda = ar_tilda.transpose([2, 1, 0])

    print('right test:', linalg.norm(ar_tilda - A_R))

    return al_tilda, ar_tilda


"""
########################################################################################################################

MPO VUMPS Section (see Ref. [1])

No long-range interactions: W[a,a] = 0 (except for W[0,0] and W[d_w-1, d_w-1])

########################################################################################################################
"""


def create_mpo_transfer_matrix(A_L, O):
    """Returns MPO transfer matrix T_O."""

    T_O = ncon([A_L, O, np.conj(A_L)],
               [[-4, 1, -2], [1, 2], [-3, 2, -1]])
    return T_O


def create_LW(A_L, C, W):
    """Returns left fixed point of MPO and the energy expectation value."""

    d_w = W.shape[0]
    dim, d = A_L.shape[:2]

    L_W = np.zeros([d_w, dim, dim], dtype=complex)
    L_W[d_w - 1] = np.eye(dim, dim)
    for i in reversed(range(d_w - 1)):  # dw-2, dw-3, ..., 1, 0
        for j in range(i + 1, d_w):  # j > i: i+1, ..., d_w-1
            L_W[i] += ncon([L_W[j], create_mpo_transfer_matrix(A_L, W[j, i])],
                           [[1, 2], [-1, -2, 1, 2]])  # Lw[i] = Lw[j] T[j,i], see Eq. (C17)

    C_r = C.T
    R = ncon([np.conj(C_r), C_r],
             [[1, -1], [1, -2]])
    e_Lw = ncon([R, L_W[0]],
                [[1, 2], [1, 2]])  # Eq. (C27) in Ref. [1]
    L_W[0] -= e_Lw * np.eye(dim, dim)
    # L_W[0] = sum_right_left(L_W[0], A_L, C_r)
    L_W[0] = half_infinity_sum_explicit(L_W[0], A_L, C_r)  # Eq. (C25a) in Ref. [1]
    L_W = L_W.transpose([1, 0, 2])

    return L_W, e_Lw


def create_RW(A_R, C, W):
    """Returns right fixed point of MPO and the energy expectation value."""

    d_w = W.shape[0]
    dim, d = A_R.shape[:2]

    R_W = np.zeros([d_w, dim, dim], dtype=complex)
    R_W[0] = np.eye(dim, dim)
    for i in range(1, d_w):  # 1,2,...,dw-1
        for j in reversed(range(i)):  # j < i: i-1,i-2,...,0
            R_W[i] += ncon([R_W[j], create_mpo_transfer_matrix(A_R, W[i, j])],
                           [[1, 2], [-1, -2, 1, 2]])  # Rw[i] = T[i,j] R[j], see Eq. (C18)

    L = ncon([np.conj(C), C],
             [[1, -1], [1, -2]])
    e_Rw = ncon([L, R_W[d_w-1]],
                [[1, 2], [1, 2]])  # Eq. (C27) in Ref. [1]
    R_W[d_w - 1] -= e_Rw * np.eye(dim, dim)
    # R_W[d_w-1] = sum_right_left(R_W[d_w-1], A_R, C)
    R_W[d_w-1] = half_infinity_sum_explicit(R_W[d_w-1], A_R, C)  # Eq. (C25b) in Ref. [1]
    R_W = R_W.transpose([1, 0, 2])

    return R_W, e_Rw


def get_Lh_Rh_mpo(A_L, A_R, C, W):
    """Returns left and right fixed points of MPO and the energy expectation value."""

    # TODO: assert lower-triangular form of W
    # TODO: assert W[a,a] = 0 (except for W[0,0] and W[d_w-1, d_w-1])

    L_W, e_Lw = create_LW(A_L, C, W)
    R_W, e_Rw = create_RW(A_R, C, W)

    return L_W, R_W, (e_Lw + e_Rw) / 2


def vumps_mpo(W, A, eta=1e-8):

    print('>' * 100)
    print('VUMPS for MPO...')

    def map_Hac(Ac):
        Ac = Ac.reshape(dim, d, dim)
        # e_eye = energy * np.eye(d ** 2, d ** 2).reshape(d, d, d, d)
        Ac_new = ncon([L_W, Ac, W, R_W],
                      [[-1, 3, 1], [1, 5, 2], [3, 4, 5, -2], [-3, 4, 2]])
        return Ac_new.reshape(-1)

    def map_Hc(C):
        C = C.reshape(dim, dim)
        C_new = ncon([L_W, C, R_W],
                     [[-1, 3, 1], [1, 2], [-2, 3, 2]])
        return C_new.reshape(-1)

    dim, d, _ = A.shape

    lam, gamma = umps_to_lam_gamma_form(A)
    lam, gamma = lam_gamma_to_canonical(lam, gamma)
    A_L, A_R = canonical_to_left_and_right_canonical(lam, gamma)

    C = lam
    Ac = ncon([A_L, C],
              [[-1, -2, 1], [1, -3]])
    delta = eta * 1000

    energy = 0
    energy_mem = -1
    count = 0

    while (delta > eta and abs(energy - energy_mem) > eta / 10) or count < 15:

        energy_mem = energy
        L_W, R_W, energy = get_Lh_Rh_mpo(A_L, A_R, C, W)

        E_Ac, Ac = eigs(LinearOperator((dim ** 2 * d, dim ** 2 * d), matvec=map_Hac), k=1, which='SR',
                        v0=Ac.reshape(-1), tol=delta / 10)

        Ac = Ac.reshape(dim, d, dim)

        E_C, C = eigs(LinearOperator((dim ** 2, dim ** 2), matvec=map_Hc), k=1, which='SR',
                      v0=C.reshape(-1), tol=delta / 10)

        C = C.reshape(dim, dim)

        A_L, A_R = min_Ac_C(Ac, C)
        Al_C = ncon([A_L, C],
                    [[-1, -2, 1], [1, -3]])
        delta = linalg.norm(Ac - Al_C)

        if count % 5 == 0:
            print(50 * '-' + 'steps', count, 50 * '-')
            print('energy = ', energy)
            print('delta = ', delta)
            print('Eac = ', E_Ac)
            print('Ec = ', E_C)
            print('Eac-Ec = ', E_Ac-E_C)
            print('Eac/Ec = ', E_Ac/E_C)

        count += 1

    print(50 * '-' + ' final ' + 50 * '-')
    print('energy = ', energy)
    energy_error = abs((energy - Exact) / Exact)
    print('Error', energy_error)
    # test_energy = ncon([L_W, Ac, W, np.conj(Ac), R_W],
    #                    [[3, 2, 1], [1, 7, 4], [2, 5, 7, 8], [3, 8, 6], [6, 5, 4]])
    # print('test_energy = ', test_energy)

    return energy, Ac, C, A_L, A_R, L_W, R_W


"""
# XXZ model
q = 0.1
h_loc = np.kron(SX, SX) + np.kron(SY, SY) + q * np.kron(SZ, SZ)
h_loc = np.real(h_loc).reshape(2, 2, 2, 2)
"""

# TFI model
hz_field = 0.49
h_loc = (-np.kron(SX, SX) - (hz_field / 2) * (np.kron(SZ, np.eye(2)) + np.kron(np.eye(2), SZ)))
h_loc = h_loc.reshape(2, 2, 2, 2)

N = 1000000
x = np.linspace(0, 2 * np.pi, N + 1)
y = np.sqrt((hz_field - 1) ** 2 + 4 * hz_field * np.sin(x / 2) ** 2)
exact = -0.5 * sum(y[1:(N + 1)] + y[:N]) / N

print('exact', exact)

# print(h_loc)

if __name__ == '__main__':

    dim = 25  # virtual bond dimension
    d = 2  # physical dimension
    # seed = 1234
    # np.random.seed(seed)
    eta = 1e-7

    A = np.random.rand(dim, d, dim)
    l0 = np.random.randn(dim, dim)
    r0 = np.random.randn(dim, dim)

    lam, gamma = umps_to_lam_gamma_form(A)
    lam, gamma = lam_gamma_to_canonical(lam, gamma)
    A_L, A_R = canonical_to_left_and_right_canonical(lam, gamma)
    C = lam
    Ac = ncon([A_L, C],
              [[-1, -2, 1], [1, -3]])

    # TODO: verify tensor properties

    e, e_mem = 0, 1
    num_of_iter = 0
    delta = eta * 10

    while num_of_iter < 10 or (delta > eta or abs(e - e_mem) > eta / 10):

        e_mem = e
        e = evaluate_energy_two_sites(A_L, A_R, Ac, h_loc)
        print('iter: ', num_of_iter, 'energy: ', e, 'abs error: ', abs(e - exact))

        e_eye = e * np.eye(d ** 2, d ** 2).reshape(d, d, d, d)
        h_L = create_h_l(A_L, h_loc)
        h_tilda = h_loc - e_eye

        C_r = C.T  # TODO: check this

        L_h = half_infinity_sum_explicit(h_L, A_L, C_r, eps_sum=delta / 10)
        # L_h = half_infinity_sum_explicit(h_L, A_L, C_r, eps_sum=1.E-10)
        h_L = create_h_l(A_L, h_tilda)
        # L_h_iter = half_infinity_sum_iterative(h_L, A_L, eps_sum=1.E-10)

        # print('L_h')
        # print(L_h)
        # print('L_h_iter')
        # print(L_h_iter)

        # print('relative difference:', linalg.norm(L_h - L_h_iter) / linalg.norm(L_h))

        h_R = create_h_r(A_R, h_loc)

        R_h = half_infinity_sum_explicit(h_R, A_R, C, eps_sum=delta / 10)
        # R_h = half_infinity_sum_explicit(h_R, A_R, C, eps_sum=1.E-10)
        h_R = create_h_r(A_R, h_tilda)
        # R_h_iter = half_infinity_sum_iterative(h_R, A_R, eps_sum=1.E-10)

        # print('R_h')
        # print(R_h)
        # print('R_h_iter')
        # print(R_h_iter)

        # print('relative difference:', linalg.norm(R_h - R_h_iter) / linalg.norm(R_h))
        # print(L_h - L_h_iter)

        map_Hac = create_Hac_map(A_L, A_R, L_h, R_h, h_loc)
        map_Hc = create_Hc_map(A_L, A_R, L_h, R_h, h_loc)

        linear_system = LinearOperator((dim ** 2 * d, dim ** 2 * d), matvec=map_Hac)
        E_Ac, Ac = eigs(linear_system, k=1, which='SR', v0=Ac.reshape(-1), tol=delta / 10)
        # E_Ac, Ac = eigsh(linear_system, k=1, which='SA', v0=Ac.reshape(-1), tol=delta / 10)

        print('E_Ac', E_Ac)

        Ac = Ac.reshape(dim, d, dim)

        linear_system = LinearOperator((dim ** 2, dim ** 2), matvec=map_Hc)
        E_C, C = eigs(linear_system, k=1, which='SR', v0=C.reshape(-1), tol=delta / 10)
        # E_C, C = eigsh(linear_system, k=1, which='SA', v0=C.reshape(-1), tol=delta / 10)

        print('E_C', E_C)

        C = C.reshape(dim, dim)

        A_L, A_R = min_Ac_C(Ac, C)

        Al_C = ncon([A_L, C],
                    [[-1, -2, 1], [1, -3]])

        delta = linalg.norm(Ac - Al_C)  # norm of the gradient
        print('delta', delta)
        num_of_iter += 1
