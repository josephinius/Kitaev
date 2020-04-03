import unittest
import numpy as np
import constants


class TestConstants(unittest.TestCase):

    def test_get_spin_operators(self):

        spin_to_ops = {
            '1/2': (constants.sx12, constants.sy12, constants.sz12, np.eye(2)),
            '1': (constants.SX1, constants.SY1, constants.SZ1, np.eye(3)),
            '3/2': (constants.sx32, constants.sy32, constants.sz32, np.eye(4)),
            '2': (constants.SX2, constants.SY2, constants.SZ2, np.eye(5)),
            '5/2': (constants.sx52, constants.sy52, constants.sz52, np.eye(6)),
            '3': (constants.SX3, constants.SY3, constants.SZ3, np.eye(7))
        }

        for spin in ('1/2', '1', '3/2', '2', '5/2', '3'):
            for o1, o2 in zip(constants.get_spin_operators(spin), spin_to_ops[spin]):
                self.assertTrue(np.allclose(o1, o2, atol=1.E-16))


if __name__ == '__main__':
    unittest.main()
