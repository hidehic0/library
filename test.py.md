---
data:
  _extendedDependsOn: []
  _extendedRequiredBy: []
  _extendedVerifiedWith: []
  _isVerificationFailed: false
  _pathExtension: py
  _verificationStatusIcon: ':warning:'
  attributes:
    links: []
  bundledCode: "Traceback (most recent call last):\n  File \"/opt/hostedtoolcache/Python/3.13.2/x64/lib/python3.13/site-packages/onlinejudge_verify/documentation/build.py\"\
    , line 71, in _render_source_code_stat\n    bundled_code = language.bundle(stat.path,\
    \ basedir=basedir, options={'include_paths': [basedir]}).decode()\n          \
    \         ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\
    \  File \"/opt/hostedtoolcache/Python/3.13.2/x64/lib/python3.13/site-packages/onlinejudge_verify/languages/python.py\"\
    , line 96, in bundle\n    raise NotImplementedError\nNotImplementedError\n"
  code: "import unittest\nfrom libs.grid import coordinate_check, grid_moves\nfrom\
    \ libs.math_func import is_prime, simple_sigma, factorization_plural\nfrom libs.utils\
    \ import lowerlist, upperlist, INF\nfrom libs.modint import mod_add, mod_sub\n\
    from libs.coordinate_compression import coordinate_compression\n\n\nclass GridTests(unittest.TestCase):\n\
    \    def test_coordinate_check(self):\n        test_cases = [\n            (0,\
    \ 0, 1, 1, True),\n            (1, 3, 1, 1, False),\n            (3, 0, 4, 10,\
    \ True),\n            (5, 1, 6, 4, True),\n        ]\n\n        for x, y, h, w,\
    \ ans in test_cases:\n            with self.subTest(x=x, y=y, h=h, w=w):\n   \
    \             self.assertEqual(coordinate_check(x, y, h, w), ans)\n\n\nclass TestMathFunctions(unittest.TestCase):\n\
    \    def test_is_prime(self):\n        test_cases = [\n            (1, False),\n\
    \            (2, True),\n            (3, True),\n            (4, False),\n   \
    \         (5, True),\n            (6, False),\n            (1747, True),\n   \
    \         (256, False),\n            (4559258600391305197, True),\n          \
    \  (4870395676773890257, True),\n            (2696910492149275857, False),\n \
    \       ]\n\n        for i, ans in test_cases:\n            with self.subTest(i=i):\n\
    \                self.assertEqual(is_prime(i), ans)\n\n    def test_simple_sigma(self):\n\
    \        test_cases = [\n            (3, 6),\n            (5000000000, 12500000002500000000),\n\
    \            (2000000000, 2000000001000000000),\n            (3090419468, 4775346245641911246),\n\
    \        ]\n\n        for i, ans in test_cases:\n            with self.subTest(i=i):\n\
    \                self.assertEqual(simple_sigma(i), ans)\n\n    def test_factorization_plural(self):\n\
    \        test_cases = [([5, 10], [[[5, 1]], [[2, 1], [5, 1]]])]\n\n        for\
    \ l, ans in test_cases:\n            with self.subTest(l=l):\n               \
    \ self.assertEqual(factorization_plural(l), ans)\n\n\nclass TestUtilsVariable(unittest.TestCase):\n\
    \    def test_lowerlist(self):\n        self.assertEqual([chr(i) for i in range(97,\
    \ 123)], lowerlist)\n\n    def test_upperlist(self):\n        self.assertEqual([chr(i)\
    \ for i in range(65, 91)], upperlist)\n\n    def test_inf_value(self):\n     \
    \   self.assertEqual(1 << 63, INF)\n\n\nclass TestModFunctions(unittest.TestCase):\n\
    \    def test_modadd(self):\n        test_cases = [(1, 1, 4, 2), (5, 5, 7, 3),\
    \ (10, -5, 3, 2)]\n\n        for a, b, mod, ans in test_cases:\n            with\
    \ self.subTest(a=a, b=b, mod=mod):\n                self.assertEqual(mod_add(a,\
    \ b, mod), ans)\n\n    def test_modsub(self):\n        test_cases = [(3, 1, 4,\
    \ 2), (1, 5, 3, 2), (15, 3, 2, 0)]\n\n        for a, b, mod, ans in test_cases:\n\
    \            with self.subTest(a=a, b=b, mod=mod):\n                self.assertEqual(mod_sub(a,\
    \ b, mod), ans)\n\n\nclass TestCoordinateCompression(unittest.TestCase):\n   \
    \ def test_coordinate_compression(self):\n        test_cases = [\n           \
    \ ((8, 100, 33, 12, 6, 1211), [1, 4, 3, 2, 0, 5]),\n            ((5, 5, 5, 5,\
    \ 5, 5), [0, 0, 0, 0, 0, 0]),\n        ]\n\n        for lis, ans in test_cases:\n\
    \            with self.subTest(lis=lis):\n                self.assertEqual(coordinate_compression(lis),\
    \ ans)\n\n\nif __name__ == \"__main__\":\n    unittest.main()\n"
  dependsOn: []
  isVerificationFile: false
  path: test.py
  requiredBy: []
  timestamp: '1970-01-01 00:00:00+00:00'
  verificationStatus: LIBRARY_NO_TESTS
  verifiedWith: []
documentation_of: test.py
layout: document
redirect_from:
- /library/test.py
- /library/test.py.html
title: test.py
---
