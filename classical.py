import numpy as np
import ctmrg
import math
from math import cos

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


ising_ham_tuple = (hamiltonian_ising_weight, hamiltonian_ising_tm, hamiltonian_ising_corner)


# Potts model


def delta_function(i, j):
    return int(i == j)


def hamiltonian_potts_weight(*s, h=.0, h_direction=0):
    s1, s2, s3, s4 = s
    interaction_terms = - (delta_function(s1, s2) + delta_function(s2, s3) +
                           delta_function(s3, s4) + delta_function(s4, s1))
    global_field_terms = - h * (delta_function(s1, h_direction) + delta_function(s2, h_direction) +
                                delta_function(s3, h_direction) + delta_function(s4, h_direction)) / 2
    return interaction_terms + global_field_terms


def hamiltonian_potts_tm(*s, h=.0, g=.0, h_direction=0, g_direction=0):
    s1, s2, s3, s4 = s
    interaction_terms = - (delta_function(s1, s2) + delta_function(s2, s3) +
                           delta_function(s3, s4) + delta_function(s4, s1))
    global_field_terms = - h * (2 * delta_function(s1, h_direction) + delta_function(s2, h_direction) +
                                delta_function(s3, h_direction) + delta_function(s4, h_direction)) / 2
    boundary_field_terms = - g * delta_function(s1, g_direction)
    return interaction_terms + global_field_terms + boundary_field_terms


def hamiltonian_potts_corner(*s, h=.0, g=.0, h_direction=0, g_direction=0):
    s1, s2, s3, s4 = s
    interaction_terms = - (delta_function(s1, s2) + delta_function(s2, s3) +
                           delta_function(s3, s4) + delta_function(s4, s1))
    global_field_terms = - h * (2 * delta_function(s1, h_direction) + delta_function(s2, h_direction) +
                                delta_function(s3, h_direction) + 2 * delta_function(s4, h_direction)) / 2
    boundary_field_terms = - g * (delta_function(s1, g_direction) + delta_function(s4, g_direction))
    return interaction_terms + global_field_terms + boundary_field_terms


potts_ham_tuple = (hamiltonian_potts_weight, hamiltonian_potts_tm, hamiltonian_potts_corner)


# Clock model


def hamiltonian_clock_weight(*s, h=.0, h_direction=0):  # s1, s2, s3, s4 = s
    z1, z2, z3, z4, h_direction = map(lambda x: 2 * math.pi * x / Q, s + (h_direction,))
    interaction_terms = - (cos(z1 - z2) + cos(z2 - z3) + cos(z3 - z4) + cos(z4 - z1))
    global_field_terms = - h * (cos(z1 - h_direction) + cos(z2 - h_direction) +
                                cos(z3 - h_direction) + cos(z4 - h_direction)) / 2
    return interaction_terms + global_field_terms


def hamiltonian_clock_tm(*s, h=.0, g=.0, h_direction=0, g_direction=0):  # s1, s2, s3, s4 = s
    z1, z2, z3, z4, h_direction, g_direction = map(lambda x: 2 * math.pi * x / Q, s + (h_direction, g_direction))
    interaction_terms = - (cos(z1 - z2) + cos(z2 - z3) + cos(z3 - z4) + cos(z4 - z1))
    global_field_terms = - h * (2 * cos(z1 - h_direction) + cos(z2 - h_direction) +
                                cos(z3 - h_direction) + cos(z4 - h_direction)) / 2
    boundary_field_terms = - g * cos(z1 - g_direction)
    return interaction_terms + global_field_terms + boundary_field_terms


def hamiltonian_clock_corner(*s, h=.0, g=.0, h_direction=0, g_direction=0):  # s1, s2, s3, s4 = s
    z1, z2, z3, z4, h_direction, g_direction = map(lambda x: 2 * math.pi * x / Q, s + (h_direction, g_direction))
    interaction_terms = - (cos(z1 - z2) + cos(z2 - z3) + cos(z3 - z4) + cos(z4 - z1))
    global_field_terms = - h * (2 * cos(z1 - h_direction) + cos(z2 - h_direction) +
                                cos(z3 - h_direction) + 2 * cos(z4 - h_direction)) / 2
    boundary_field_terms = - g * (cos(z1 - g_direction) + cos(z4 - g_direction))
    return interaction_terms + global_field_terms + boundary_field_terms


clock_ham_tuple = (hamiltonian_clock_weight, hamiltonian_clock_tm, hamiltonian_clock_corner)


# Model of social influence


def social_interaction(s1, s2):

    s1a, s1b = divmod(s1, Q)
    s2a, s2b = divmod(s2, Q)

    t1a, t1b, t2a, t2b = map(lambda x: 2 * math.pi * x / Q, (s1a, s1b, s2a, s2b))

    return - delta_function(s1a, s2a) * cos(t1b - t2b) - delta_function(s1b, s2b) * cos(t1a - t2a)


def hamiltonian_social_weight(*s, h=.0, h_direction=0):
    """

    Case of two features F=2 of Q traits each.

    (s1a, s1b), (s2a, s2b), (s3a, s3b), (s4a, s4b) = s

    """

    # TODO: generalize for multiple features F > 2

    s1, s2, s3, s4 = s

    interaction_terms = (social_interaction(s1, s2) + social_interaction(s2, s3) +
                         social_interaction(s3, s4) + social_interaction(s4, s1))

    # Obviously, there are many reasonable ways how to introduce the global field.
    global_field_terms = h * (social_interaction(s1, h_direction) + social_interaction(s2, h_direction) +
                              social_interaction(s3, h_direction) + social_interaction(s4, h_direction)) / 2

    return interaction_terms + global_field_terms


def hamiltonian_social_tm(*s, h=.0, g=.0, h_direction=0, g_direction=0):
    """

    Case of two features F=2 of Q traits each.

    (s1a, s1b), (s2a, s2b), (s3a, s3b), (s4a, s4b) = s

    """

    s1, s2, s3, s4 = s

    interaction_terms = (social_interaction(s1, s2) + social_interaction(s2, s3) +
                         social_interaction(s3, s4) + social_interaction(s4, s1))

    global_field_terms = h * (2 * social_interaction(s1, h_direction) + social_interaction(s2, h_direction) +
                              social_interaction(s3, h_direction) + social_interaction(s4, h_direction)) / 2

    boundary_field_terms = g * social_interaction(s1, g_direction) / 2

    return interaction_terms + global_field_terms + boundary_field_terms


def hamiltonian_social_corner(*s, h=.0, g=.0, h_direction=0, g_direction=0):
    """

    Case of two features F=2 of Q traits each.

    (s1a, s1b), (s2a, s2b), (s3a, s3b), (s4a, s4b) = s

    """

    s1, s2, s3, s4 = s

    interaction_terms = (social_interaction(s1, s2) + social_interaction(s2, s3) +
                         social_interaction(s3, s4) + social_interaction(s4, s1))

    global_field_terms = h * (2 * social_interaction(s1, h_direction) + social_interaction(s2, h_direction) +
                              social_interaction(s3, h_direction) + 2 * social_interaction(s4, h_direction)) / 2

    boundary_field_terms = g * (social_interaction(s1, g_direction) + social_interaction(s4, g_direction))

    return interaction_terms + global_field_terms + boundary_field_terms


social_ham_tuple = (hamiltonian_social_weight, hamiltonian_social_tm, hamiltonian_social_corner)


def initialization(hamiltonian, temperature, h, g):

    weight_ham, tm_ham, corner_ham = hamiltonian

    qf = Q ** F

    w = np.zeros((qf, qf, qf, qf))
    tm = np.zeros((qf, qf, qf))
    corner = np.zeros((qf, qf))

    for i in range(qf):
        for j in range(qf):
            for k in range(qf):
                for l in range(qf):
                    w[i, j, k, l] = np.exp(-weight_ham(i, j, k, l, h=h) / temperature)
                    tm[j, k, l] += np.exp(-tm_ham(i, j, k, l, h=h, g=g) / temperature)
                    corner[j, k] += np.exp(-corner_ham(i, j, k, l, h=h, g=g) / temperature)

    return w, tm, corner


# Initialization...

file_name = 'classical.txt'

dim = 50

model = "Social"  # "Ising", "Potts", "Clock", (later also "Social")
Q = 2
temperature = .1
h = .0
g = 1.E-10

# setting number of "features" F
if model == "Social":
    F = 2
else:
    F = 1

if model == "Ising":
    hamiltonian = ising_ham_tuple
elif model == "Potts":
    hamiltonian = potts_ham_tuple
elif model == "Clock":
    hamiltonian = clock_ham_tuple
elif model == "Social":
    hamiltonian = social_ham_tuple

with open(file_name, 'w') as f:
    f.write(f'# dim={dim}, h={h}, g={g}\n')

"""
# free energy calculation test
w, tm, corner = initialization(hamiltonian, temperature, h, g)
weight = w
tms = [tm[:] for _ in range(4)]
corners = [corner[:] for _ in range(4)]
weight_imp = w
ctm = ctmrg.CTMRG(dim, weight, corners, tms, weight_imp, algorithm='Orus')
ctm.temperature = temperature
ctm.ctmrg_iteration(100_000)
exit()
"""

for temperature in np.linspace(2.0, 3.0, num=100, endpoint=False):

    w, tm, corner = initialization(hamiltonian, temperature, h, g)

    weight = w
    tms = [tm[:] for _ in range(4)]
    corners = [corner[:] for _ in range(4)]
    weight_imp = w  # not an actual impurity - just for testing

    ctm = ctmrg.CTMRG(dim, weight, corners, tms, weight_imp, algorithm='Corboz')
    ctm.temperature = temperature
    ctm.ctmrg_iteration(1000)
    free_energy = ctm.classical_free_energy
    fast_free_energy = ctm.fast_free_energy
    num_of_iter = ctm.iter_counter
    with open(file_name, 'a') as f:
        f.write('%.15f\t%.15f\t%.15f\t%d\n' % (temperature, free_energy, fast_free_energy, num_of_iter))
