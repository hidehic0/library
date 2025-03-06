# competitive-verifier: UNITTEST PYTHON_UNITTEST_RESULT
import unittest
from libs.grid import coordinate_check, grid_moves
from libs.math_func import is_prime, simple_sigma, factorization_plural
from libs.utils import lowerlist, upperlist, INF
from libs.modint import mod_add, mod_sub
from libs.coordinate_compression import coordinate_compression


class GridTests(unittest.TestCase):
    def test_coordinate_check(self):
        test_cases = [
            (0, 0, 1, 1, True),
            (1, 3, 1, 1, False),
            (3, 0, 4, 10, True),
            (5, 1, 6, 4, True),
        ]

        for x, y, h, w, ans in test_cases:
            with self.subTest(x=x, y=y, h=h, w=w):
                self.assertEqual(coordinate_check(x, y, h, w), ans)


class TestMathFunctions(unittest.TestCase):
    def test_is_prime(self):
        test_cases = [
            (1, False),
            (2, True),
            (3, True),
            (4, False),
            (5, True),
            (6, False),
            (1747, True),
            (256, False),
            (4559258600391305197, True),
            (4870395676773890257, True),
            (2696910492149275857, False),
        ]

        for i, ans in test_cases:
            with self.subTest(i=i):
                self.assertEqual(is_prime(i), ans)

    def test_simple_sigma(self):
        test_cases = [
            (3, 6),
            (5000000000, 12500000002500000000),
            (2000000000, 2000000001000000000),
            (3090419468, 4775346245641911246),
        ]

        for i, ans in test_cases:
            with self.subTest(i=i):
                self.assertEqual(simple_sigma(i), ans)

    def test_factorization_plural(self):
        test_cases = [([5, 10], [[[5, 1]], [[2, 1], [5, 1]]])]

        for l, ans in test_cases:
            with self.subTest(l=l):
                self.assertEqual(factorization_plural(l), ans)


class TestUtilsVariable(unittest.TestCase):
    def test_lowerlist(self):
        self.assertEqual([chr(i) for i in range(97, 123)], lowerlist)

    def test_upperlist(self):
        self.assertEqual([chr(i) for i in range(65, 91)], upperlist)

    def test_inf_value(self):
        self.assertEqual(1 << 63, INF)


class TestModFunctions(unittest.TestCase):
    def test_modadd(self):
        test_cases = [(1, 1, 4, 2), (5, 5, 7, 3), (10, -5, 3, 2)]

        for a, b, mod, ans in test_cases:
            with self.subTest(a=a, b=b, mod=mod):
                self.assertEqual(mod_add(a, b, mod), ans)

    def test_modsub(self):
        test_cases = [(3, 1, 4, 2), (1, 5, 3, 2), (15, 3, 2, 0)]

        for a, b, mod, ans in test_cases:
            with self.subTest(a=a, b=b, mod=mod):
                self.assertEqual(mod_sub(a, b, mod), ans)


class TestCoordinateCompression(unittest.TestCase):
    def test_coordinate_compression(self):
        test_cases = [
            ((8, 100, 33, 12, 6, 1211), [1, 4, 3, 2, 0, 5]),
            ((5, 5, 5, 5, 5, 5), [0, 0, 0, 0, 0, 0]),
        ]

        for lis, ans in test_cases:
            with self.subTest(lis=lis):
                self.assertEqual(coordinate_compression(lis), ans)


if __name__ == "__main__":
    unittest.main()
