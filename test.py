import unittest
from libs.grid import coordinate_check, grid_moves
from libs.math_func import is_prime


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


if __name__ == "__main__":
    unittest.main()
