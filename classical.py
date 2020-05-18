import math
import numpy as np
import ctmrg

# Ising model


def hamiltonian_ising_weight(*s, h=.0):  # s1, s2, s3, s4 = s
    # h - global field
    z1, z2, z3, z4 = map(lambda x: 2 * x - 1, s)
    return - (z1 * z2 + z2 * z3 + z3 * z4 + z4 * z1) - h * (z1 + z2 + z3 + z4) / 2


def hamiltonian_ising_tm(*s, h=.0, g=.0):  # s1, s2, s3, s4 = s
    # h - global field
    # g - boundary field
    z1, z2, z3, z4 = map(lambda x: 2 * x - 1, s)
    return - (z1 * z2 + z2 * z3 + z3 * z4 + z4 * z1) - h * (2 * z1 + z2 + z3 + z4) / 2 - g * z1


def hamiltonian_ising_corner(*s, h=.0, g=.0):  # s1, s2, s3, s4 = s
    # h - global field
    # g - boundary field
    z1, z2, z3, z4 = map(lambda x: 2 * x - 1, s)
    return - (z1 * z2 + z2 * z3 + z3 * z4 + z4 * z1) - h * (2 * z1 + z2 + z3 + 2 * z4) / 2 - g * (z1 + z4)


# Potts model


def delta_function(i, j):
    return int(i == j)


def hamiltonian_potts_weight(*s, h=.0, field_direction=0):
    s1, s2, s3, s4 = s
    interaction_terms = - (delta_function(s1, s2) +
                           delta_function(s2, s3) +
                           delta_function(s3, s4) +
                           delta_function(s4, s1))
    field_terms = - h * (delta_function(s1, field_direction) +
                         delta_function(s2, field_direction) +
                         delta_function(s3, field_direction) +
                         delta_function(s4, field_direction)) / 2
    return interaction_terms + field_terms


def hamiltonian_potts_tm(*s, h=.0, g=.0, h_direction=0, g_direction=0):
    s1, s2, s3, s4 = s
    interaction_terms = - (delta_function(s1, s2) +
                           delta_function(s2, s3) +
                           delta_function(s3, s4) +
                           delta_function(s4, s1))
    global_field_terms = - h * (2 * delta_function(s1, h_direction) +
                                delta_function(s2, h_direction) +
                                delta_function(s3, h_direction) +
                                delta_function(s4, h_direction)) / 2
    boundary_field_terms = - g * delta_function(s1, g_direction)
    return interaction_terms + global_field_terms + boundary_field_terms


def hamiltonian_potts_corner(*s, h=.0, g=.0, h_direction=0, g_direction=0):
    s1, s2, s3, s4 = s
    interaction_terms = - (delta_function(s1, s2) +
                           delta_function(s2, s3) +
                           delta_function(s3, s4) +
                           delta_function(s4, s1))
    global_field_terms = - h * (2 * delta_function(s1, h_direction) +
                                delta_function(s2, h_direction) +
                                delta_function(s3, h_direction) +
                                2 * delta_function(s4, h_direction)) / 2
    boundary_field_terms = - g * (delta_function(s1, g_direction) + delta_function(s4, g_direction))
    return interaction_terms + global_field_terms + boundary_field_terms


def initialization(q, temperature, h, g):

    w = np.zeros((q, q, q, q))
    tm = np.zeros((q, q, q))
    corner = np.zeros((q, q))

    for i in range(q):
        for j in range(q):
            for k in range(q):
                for l in range(q):
                    w[i, j, k, l] = np.exp(-hamiltonian_ising_weight(i, j, k, l, h=h) / temperature)
                    tm[j, k, l] += np.exp(-hamiltonian_ising_tm(i, j, k, l, h=h, g=g) / temperature)
                    corner[j, k] += np.exp(-hamiltonian_ising_corner(i, j, k, l, h=h, g=g) / temperature)

    return w, tm, corner


# TODO: Clock model
# TODO: Model of social influence


# Initialization...

file_name = 'classical.txt'

dim = 8

q = 2
temperature = .1
h = .0
g = .0

with open(file_name, 'w') as f:
    f.write(f'# dim={dim}, h={h}, g={g}\n')

for temperature in np.linspace(2.0, 2.5, num=50, endpoint=False):

    w, tm, corner = initialization(q, temperature, h, g)

    weight = w
    tms = [tm[:] for _ in range(4)]
    corners = [corner[:] for _ in range(4)]
    weight_imp = w

    ctm = ctmrg.CTMRG(dim, weight, corners, tms, weight_imp, algorithm='Orus')
    ctm.temperature = temperature
    ctm.ctmrg_iteration(10_000)
    fast_free_energy = ctm.fast_free_energy
    num_of_iter = ctm.iter_counter
    with open(file_name, 'a') as f:
        f.write('%.15f\t%.15f\t%d\n' % (temperature, fast_free_energy, num_of_iter))























