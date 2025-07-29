# competitive-verifier: UNITTEST PYTHON_UNITTEST_RESULT
import unittest

from libs.coordinate_compression import compress_1d
from libs.grid import coordinate_check
from libs.heap import HeapMin, _keys_for_heapq
from libs.math_func import factorization_plural, is_prime, simple_sigma
from libs.modint import mod_add, mod_sub
from libs.rerooting import rerooting
from libs.utils import INF, lowerlist, upperlist


class GridTests(unittest.TestCase):
    def test_coordinate_check(self) -> None:
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
    def test_is_prime(self) -> None:
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

    def test_simple_sigma(self) -> None:
        test_cases = [
            (3, 6),
            (5000000000, 12500000002500000000),
            (2000000000, 2000000001000000000),
            (3090419468, 4775346245641911246),
        ]

        for i, ans in test_cases:
            with self.subTest(i=i):
                self.assertEqual(simple_sigma(i), ans)

    def test_factorization_plural(self) -> None:
        test_cases = [([5, 10], [[[5, 1]], [[2, 1], [5, 1]]])]

        for l, ans in test_cases:
            with self.subTest(l=l):
                self.assertEqual(factorization_plural(l), ans)


class TestUtilsVariable(unittest.TestCase):
    def test_lowerlist(self) -> None:
        self.assertEqual([chr(i) for i in range(97, 123)], lowerlist)

    def test_upperlist(self) -> None:
        self.assertEqual([chr(i) for i in range(65, 91)], upperlist)

    def test_inf_value(self) -> None:
        self.assertEqual(1 << 63, INF)


class TestModFunctions(unittest.TestCase):
    def test_modadd(self) -> None:
        test_cases = [(1, 1, 4, 2), (5, 5, 7, 3), (10, -5, 3, 2)]

        for a, b, mod, ans in test_cases:
            with self.subTest(a=a, b=b, mod=mod):
                self.assertEqual(mod_add(a, b, mod), ans)

    def test_modsub(self) -> None:
        test_cases = [(3, 1, 4, 2), (1, 5, 3, 2), (15, 3, 2, 0)]

        for a, b, mod, ans in test_cases:
            with self.subTest(a=a, b=b, mod=mod):
                self.assertEqual(mod_sub(a, b, mod), ans)


class TestCoordinateCompression(unittest.TestCase):
    def test_coordinate_compression(self) -> None:
        test_cases = [
            ((8, 100, 33, 12, 6, 1211), [1, 4, 3, 2, 0, 5]),
            ((5, 5, 5, 5, 5, 5), [0, 0, 0, 0, 0, 0]),
        ]

        for lis, ans in test_cases:
            with self.subTest(lis=lis):
                self.assertEqual(compress_1d(lis), ans)


class TestHeap(unittest.TestCase):
    def test_heap_key(self) -> None:
        test_cases = [((1, 2), 1), (1, 1), (((1, 3), (2, 2)), 1)]

        for lis, ans in test_cases:
            with self.subTest(lis=lis):
                self.assertEqual(_keys_for_heapq(lis), ans)

    def test_minheap(self) -> None:
        test_cases = [4, 3, 5, 1]
        L = HeapMin()

        for i in test_cases:
            L.push(i)

        self.assertTrue(len(L) == 4)
        self.assertEqual(L[0], 1)

        self.assertListEqual(
            sorted(test_cases), [L.pop() for _ in range(len(test_cases))]
        )

        test_cases = [(4, 1), (3, 2), (5, 3), (1, 4)]
        L = HeapMin()

        for i in test_cases:
            L.push(i)

        self.assertTrue(len(L) == 4)
        self.assertEqual(L[0], (1, 4))

        self.assertListEqual(
            sorted(test_cases, key=lambda x: x[0]),
            [L.pop() for _ in range(len(test_cases))],
        )


class TestReRooting(unittest.TestCase):
    def test_rerooting(self) -> None:
        G = [[] for _ in [0] * 6]

        for a, b in [[1, 2], [1, 3], [3, 4], [3, 5], [5, 6]]:
            a -= 1
            b -= 1
            G[a].append(b)
            G[b].append(a)

        res = rerooting(G, max, lambda x: x + 1, -1)
        self.assertEqual(res, [3, 4, 2, 3, 3, 4])


if __name__ == "__main__":
    unittest.main()
