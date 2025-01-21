import unittest
from libs.grid import coordinate_check, grid_moves
from libs.math_func import is_prime, simple_sigma
from libs.utils import lowerlist, upperlist, INF


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


class TestUtilsVariable(unittest.TestCase):
    def test_lowerlist(self):
        self.assertEqual([chr(i) for i in range(97, 123)], lowerlist)

    def test_upperlist(self):
        self.assertEqual([chr(i) for i in range(65, 91)], upperlist)

    def test_inf_value(self):
        self.assertEqual(1 << 63, INF)


if __name__ == "__main__":
    unittest.main()
